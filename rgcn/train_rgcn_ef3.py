"""Train an indication-only R-GCN for drug-to-disease ranking.

This experiment uses one prediction task: DRG-DIS|INDICATION, drug -> disease.
Checkpoint selection is validation micro-AUPRC for that exact task. An
auxiliary link-prediction loss (default weight 0.15) regularizes the encoder
using non-DRG-DIS training edges only. Contraindication edges are excluded
from the prediction objective and from the auxiliary pool.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from model import RGCN, build_message_graph, indication_loss, average_precision, summarize_ranks


def fmt(m):
    return (f"n={m['n']} AP={m['AUPRC']:.4f} MRR={m['MRR']:.4f} "
            f"H@1={m['H1']:.4f} H@3={m['H3']:.4f} "
            f"H@10={m['H10']:.4f} H@25={m['H25']:.4f}")


def evaluate(model, hidden, q, pos, candidates, known_q, known_pos):
    """Evaluate one positive set per drug against the full disease candidate set."""
    empty = {"n": 0, "AUPRC": float("nan"), "MRR": float("nan"),
             "H1": float("nan"), "H3": float("nan"), "H10": float("nan"), "H25": float("nan")}
    if q.numel() == 0:
        return empty
    uq, inverse = torch.unique(q, sorted=True, return_inverse=True)
    scores = model.score_candidates(hidden, uq, 0, candidates)
    held = defaultdict(set)
    for qi, pi in zip(q.tolist(), pos.tolist()):
        held[int(qi)].add(int(pi))
    known = defaultdict(set)
    for qi, pi in zip(known_q.tolist(), known_pos.tolist()):
        known[int(qi)].add(int(pi))

    labels_all, scores_all, ranks = [], [], []
    for row, query in enumerate(uq.tolist()):
        positives = held[query]
        filt = torch.zeros(candidates.numel(), dtype=torch.bool, device=hidden.device)
        for p in known[query]:
            if p not in positives:
                filt[p] = True
        keep = ~filt
        labels = torch.zeros(candidates.numel(), dtype=torch.float32, device=hidden.device)
        labels[list(positives)] = 1.0
        labels_all.append(labels[keep])
        scores_all.append(scores[row][keep])
        for p in positives:
            rank_mask = filt.clone()
            rank_mask[list(positives)] = True
            rank_mask[p] = False
            valid = scores[row].masked_fill(rank_mask, float("-inf"))
            greater = (valid > scores[row, p]).sum().float()
            tied = ((valid == scores[row, p]).sum().float() - 1).clamp(min=0)
            ranks.append(1.0 + greater + 0.5 * tied)
    if not ranks:
        return empty
    rank_t = torch.stack(ranks)
    metrics = summarize_ranks(rank_t)
    metrics["H25"] = round(float((rank_t <= 25).float().mean()), 4)
    metrics["AUPRC"] = round(average_precision(torch.cat(labels_all), torch.cat(scores_all)), 4)
    metrics["n"] = int(uq.numel())
    return metrics


def train(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dev = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                       else "cpu" if args.device == "auto" else args.device)
    data = torch.load(args.split, map_location="cpu", weights_only=False)
    src, dst, rid = data["src"].long(), data["dst"].long(), data["rid"].long()
    undirected = data["undirected"].bool()
    node_type = data["node_type"].long()
    rel_names, type_names = list(data["rel_names"]), list(data["type_names"])
    n_nodes, n_rels = int(data["num_nodes"]), int(data["num_rels"])
    drug_t, dis_t = type_names.index("DRG"), type_names.index("DIS")
    ind_rids = data.get("ind_rids")
    if ind_rids is None:
        ind_rids = torch.tensor([i for i, x in enumerate(rel_names)
                                 if x.upper().startswith("DRG-DIS|") and "INDICATION" in x.upper()])
    ind_rids = ind_rids.long()
    if ind_rids.numel() == 0:
        raise SystemExit("No DRG-DIS|INDICATION relation IDs in split")

    split = {k: data["split"][k].long() for k in ("train", "val", "test")}
    drug_ids = torch.nonzero(node_type == drug_t).flatten()
    dis_ids = torch.nonzero(node_type == dis_t).flatten()
    if not drug_ids.numel() or not dis_ids.numel():
        raise SystemExit("No drug or disease candidates in graph")
    dis_lut = torch.full((n_nodes,), -1, dtype=torch.long)
    dis_lut[dis_ids] = torch.arange(dis_ids.numel())

    def indication_pairs(edge_ids):
        e = edge_ids
        take = torch.isin(rid[e], ind_rids)
        e = e[take]
        s, d = src[e], dst[e]
        forward = (node_type[s] == drug_t) & (node_type[d] == dis_t)
        backward = (node_type[d] == drug_t) & (node_type[s] == dis_t)
        ok = forward | backward
        drug = torch.where(forward[ok], s[ok], d[ok])
        disease = torch.where(forward[ok], d[ok], s[ok])
        pos = dis_lut[disease]
        if pos.numel() and (pos < 0).any():
            raise SystemExit("Indication target missing from disease candidates")
        return drug.long(), pos.long()

    pairs = {name: indication_pairs(split[name]) for name in ("train", "val", "test")}
    tr_q, tr_pos = pairs["train"]
    val_q, val_pos = pairs["val"]
    test_q, test_pos = pairs["test"]
    if not tr_q.numel() or not val_q.numel() or not test_q.numel():
        raise SystemExit(f"Empty indication split: train={tr_q.numel()} val={val_q.numel()} test={test_q.numel()}")
    known_q = torch.cat([pairs[k][0] for k in ("train", "val", "test")]).to(dev)
    known_pos = torch.cat([pairs[k][1] for k in ("train", "val", "test")]).to(dev)

    # All drug-disease relations are excluded from message passing, including
    # contraindication and held-out indication edges.
    blocked = torch.tensor([i for i, name in enumerate(rel_names)
                            if name.upper().startswith("DRG-DIS|")], dtype=torch.long)
    mp_s, mp_d, mp_r = build_message_graph(src, dst, rid, undirected, split["train"], blocked, n_rels)
    if not mp_s.numel():
        raise SystemExit("Message-passing graph is empty")
    edge_index, edge_type = torch.stack([mp_s, mp_d]).to(dev), mp_r.to(dev)
    node_type_d = node_type.to(dev)

    # Auxiliary pool: non-drug-disease training edges only. These preserve
    # general graph structure without introducing contraindication labels.
    aux_pool = split["train"][~torch.isin(rid[split["train"]], blocked)]
    type_count = torch.bincount(node_type, minlength=len(type_names))
    if aux_pool.numel():
        aux_pool = aux_pool[type_count[node_type[dst[aux_pool]]] >= 2]
    aux_s = src[aux_pool].to(dev) if aux_pool.numel() else torch.empty(0, dtype=torch.long, device=dev)
    aux_d = dst[aux_pool].to(dev) if aux_pool.numel() else torch.empty(0, dtype=torch.long, device=dev)
    aux_r = rid[aux_pool].to(dev) if aux_pool.numel() else torch.empty(0, dtype=torch.long, device=dev)
    true_keys = torch.unique((src[split["train"]] * n_nodes + dst[split["train"]]) * n_rels + rid[split["train"]]).to(dev)
    sorted_nodes = torch.argsort(node_type).to(dev)
    type_count_d = type_count.to(dev)
    type_start = torch.cumsum(type_count_d, 0) - type_count_d

    def sample_neg(tails, k):
        types = node_type_d[tails]
        counts = type_count_d[types].clamp(min=1)
        draw = (torch.rand(tails.shape[0], k, device=dev) * counts.unsqueeze(1)).long()
        draw = torch.minimum(draw, counts.unsqueeze(1) - 1)
        return sorted_nodes[type_start[types].unsqueeze(1) + draw]

    model = RGCN(n_nodes, n_rels, len(type_names), args.dim, args.layers,
                 args.dropout, args.bases).to(dev)
    conv_params = list(model.convs.parameters()) + list(model.norms.parameters())
    conv_ids = {id(p) for p in conv_params}
    other_params = [p for p in model.parameters() if id(p) not in conv_ids]
    optimizer = torch.optim.Adam([
        {"params": conv_params, "lr": args.lr * 0.1},
        {"params": other_params, "lr": args.lr},
    ])

    def encode():
        return model.encode(edge_index, edge_type, node_type_d,
                            args.edge_drop if model.training else 0.0)

    def aux_loss(hidden, batch):
        if not aux_s.numel() or args.aux <= 0:
            return hidden.new_zeros(())
        ix = torch.randint(aux_s.numel(), (batch,), device=dev)
        heads, tails, rels = aux_s[ix], aux_d[ix], aux_r[ix]
        neg = sample_neg(tails, args.neg)
        bad = torch.isin((heads[:, None] * n_nodes + neg) * n_rels + rels[:, None], true_keys)
        bad |= neg == tails[:, None]
        neg = torch.where(bad, sample_neg(tails, args.neg), neg)
        still = torch.isin((heads[:, None] * n_nodes + neg) * n_rels + rels[:, None], true_keys)
        still |= neg == tails[:, None]
        scores = torch.cat([model.score_pairs(hidden, heads, rels, tails).unsqueeze(1),
                            model.score_pairs(hidden, heads, rels, neg)], dim=1)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[:, 1:] = still
        labels = torch.zeros(batch, dtype=torch.long, device=dev)
        return indication_loss(scores, labels, mask, args.alpha, args.gamma, args.uniform)

    def run_eval(which):
        model.eval()
        with torch.no_grad():
            hidden = encode()
            q, p = pairs[which]
            result = evaluate(model, hidden, q.to(dev), p.to(dev), dis_ids.to(dev), known_q, known_pos)
        model.train()
        return result

    best = {"score": -1.0, "step": 0, "state": None, "metrics": None}
    curve, stall = [], 0
    start = time.time()
    print(f"device {dev} | nodes {n_nodes:,} | message-passing edges {mp_s.numel():,}", flush=True)
    print(f"indication positives train/val/test: {tr_q.numel()}/{val_q.numel()}/{test_q.numel()} | "
          f"candidate diseases {dis_ids.numel()} | auxiliary edges {aux_s.numel():,}", flush=True)

    for step in range(args.steps + 1):
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            val = run_eval("val")
            score = val["AUPRC"]
            curve.append({"step": step, "AUPRC": score, "metrics": val})
            improved = math.isfinite(score) and score > best["score"] + 1e-5
            if improved:
                best.update(score=score, step=step,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            metrics=val)
                stall = 0
            elif step:
                stall += 1
            print(f"[val] step {step} {fmt(val)}{' *' if improved else ''}", flush=True)
            if step and stall >= args.patience:
                print(f"Early stopping after {args.patience} evaluations without validation AUPRC improvement", flush=True)
                break
        if step == args.steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        hidden = encode()
        batch = min(args.bs, tr_q.numel())
        ix = torch.randint(tr_q.numel(), (batch,), device=dev)
        q, pos = tr_q.to(dev)[ix], tr_pos.to(dev)[ix]
        scores = model.score_candidates(hidden, q, 0, dis_ids.to(dev))
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for row, query in enumerate(q.tolist()):
            other = known_pos[known_q == query]
            if other.numel():
                mask[row, other] = True
            mask[row, pos[row]] = False
        loss = indication_loss(scores, pos, mask, args.alpha, args.gamma, args.uniform)
        loss = loss + args.aux * aux_loss(hidden, batch)
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss; skipped", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step and step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time()-start)/step:.3f}s/step", flush=True)

    if best["state"] is None:
        raise SystemExit("No valid checkpoint selected; validation AUPRC was not finite")
    model.load_state_dict(best["state"])
    result = {
        "tag": args.tag, "task": "indication_drug_to_disease", "split": "pair_random_90_5_5",
        "best_step": best["step"], "val": best["metrics"], "val_curve": curve,
        "graph": {"nodes": n_nodes, "message_passing_edges": int(mp_s.numel()),
                  "drugs": int(drug_ids.numel()), "diseases": int(dis_ids.numel()),
                  "auxiliary_edges": int(aux_s.numel())},
        "shared_gene_frac": data.get("shared_gene_frac"),
        "args": vars(args),
    }
    if args.eval_test:
        result["test"] = run_eval("test")
        print(f"[test] best step {best['step']} {fmt(result['test'])}", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "dim": args.dim, "layers": args.layers,
                "bases": args.bases, "ind_rids": ind_rids, "best_step": best["step"]},
               out / f"rgcn_{args.tag}.pt")
    with open(out / f"res_{args.tag}.json", "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print(f"Saved {out / f'res_{args.tag}.json'}", flush=True)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Indication-only drug-to-disease R-GCN experiment")
    p.add_argument("--split", default="artifacts/indication_only/okg_split.pt")
    p.add_argument("--out_dir", default="artifacts/indication_only")
    p.add_argument("--tag", default="indication_only")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--bases", type=int, default=0)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--edge_drop", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--aux", type=float, default=0.15)
    p.add_argument("--neg", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--uniform", type=float, default=0.25)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_test", action="store_true")
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
