"""Train an R-GCN for indication and contraindication link prediction.

Reports four tasks: indication/contraindication x drug->disease/disease->drug.
Each task reports AUPRC, MRR, Hits@1/3/10/25. Candidate ranking is filtered
against all known positives for that relation, except the target pair itself.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from model import RGCN, build_message_graph, indication_loss, average_precision, ranks_from_scores, summarize_ranks


def device_of(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def oriented_pairs(src, dst, rid, node_type, edge_ids, wanted_rids, drug_t, dis_t):
    if edge_ids.numel() == 0:
        z = torch.empty(0, dtype=torch.long)
        return z, z
    e = edge_ids.long()
    s, d, r = src[e], dst[e], rid[e]
    take = torch.isin(r, wanted_rids)
    s, d = s[take], d[take]
    a = (node_type[s] == drug_t) & (node_type[d] == dis_t)
    b = (node_type[d] == drug_t) & (node_type[s] == dis_t)
    ok = a | b
    return torch.where(a[ok], s[ok], d[ok]), torch.where(a[ok], d[ok], s[ok])


def make_index(drug, disease, candidates, candidate_to_pos):
    if not drug.numel():
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    q = drug
    target = disease
    pos = candidate_to_pos[target]
    if (pos < 0).any():
        raise RuntimeError("A target node is missing from the candidate list")
    return q.long(), pos.long()


def relation_score(model, hidden, query, candidates, rid):
    # Relation-specific diagonal bilinear score; unlike score_candidates in the
    # original model, this preserves the distinction between INDICATION and CONTRAINDICATION.
    h = F.normalize(hidden[query], dim=-1)
    c = F.normalize(hidden[candidates], dim=-1)
    rel = model.rel[int(rid)]
    return (h[:, None, :] * rel[None, None, :] * c[None, :, :]).sum(-1)


def known_map(query, pos):
    out = defaultdict(set)
    for q, p in zip(query.tolist(), pos.tolist()):
        out[int(q)].add(int(p))
    return out


def build_task(src, dst, rid, node_type, split, rids, drug_t, dis_t, query_direction, candidates, device):
    pairs = {}
    for name in ("train", "val", "test"):
        dr, ds = oriented_pairs(src, dst, rid, node_type, split[name], rids, drug_t, dis_t)
        if query_direction == "drug_to_disease":
            q, target = dr, ds
        else:
            q, target = ds, dr
        lookup = torch.full((int(node_type.max()) + 1 if False else int(max(int(src.max()), int(dst.max())) + 1),), -1, dtype=torch.long)
        lookup[candidates] = torch.arange(candidates.numel())
        pos = lookup[target] if target.numel() else torch.empty(0, dtype=torch.long)
        pairs[name] = (q.long(), pos.long())
    return pairs


def metric_block(model, hidden, q, pos, candidates, known_q, known_pos, rid):
    empty = {"n": 0, "AUPRC": float("nan"), "MRR": float("nan"), "H1": float("nan"),
             "H3": float("nan"), "H10": float("nan"), "H25": float("nan")}
    if q.numel() == 0:
        return empty
    uq, inverse = torch.unique(q, sorted=True, return_inverse=True)
    scores = relation_score(model, hidden, uq, candidates, rid)
    # For each query, gather all held-out positives in this split.
    positives = defaultdict(set)
    for qi, pi in zip(q.tolist(), pos.tolist()):
        positives[int(qi)].add(int(pi))
    known = known_map(known_q, known_pos)
    all_ranks, all_labels, all_scores = [], [], []
    aps = []
    for row, query in enumerate(uq.tolist()):
        mine = sorted(positives[query])
        if not mine:
            continue
        filt = torch.zeros(candidates.numel(), dtype=torch.bool, device=hidden.device)
        for p in known.get(int(query), set()):
            if p not in mine:
                filt[p] = True
        y = torch.zeros(candidates.numel(), dtype=torch.float32, device=hidden.device)
        y[mine] = 1.0
        keep = ~filt
        sc = scores[row][keep]
        yy = y[keep]
        aps.append(average_precision(yy, sc))
        all_labels.append(yy)
        all_scores.append(sc)
        for p in mine:
            true_score = scores[row, p]
            # Filter every other known positive for this query, including other
            # positives in the same held-out partition.
            rank_filter = filt.clone()
            rank_filter[mine] = True
            rank_filter[p] = False
            valid_scores = scores[row].masked_fill(rank_filter, float("-inf"))
            greater = (valid_scores > true_score).sum().float()
            tied = ((valid_scores == true_score).sum().float() - 1).clamp(min=0)
            all_ranks.append(1.0 + greater + 0.5 * tied)
    if not all_ranks:
        return empty
    ranks = torch.stack(all_ranks)
    out = summarize_ranks(ranks)
    out["H25"] = round(float((ranks <= 25).float().mean()), 4)
    out["AUPRC"] = round(average_precision(torch.cat(all_labels), torch.cat(all_scores)), 4)
    out["AUPRC_macro"] = round(float(sum(aps) / len(aps)), 4)
    return out


def fmt(b):
    if not b or not b.get("n"):
        return "n=0"
    return (f"n={b['n']} AP={b['AUPRC']:.4f} MRR={b['MRR']:.4f} "
            f"H@1={b['H1']:.4f} H@3={b['H3']:.4f} H@10={b['H10']:.4f} H@25={b['H25']:.4f}")


def train(args):
    dev = device_of(args.device)
    torch.manual_seed(args.seed)
    data = torch.load(args.split, map_location="cpu", weights_only=False)
    required = ("src", "dst", "rid", "undirected", "node_type", "split", "type_names", "rel_names")
    for k in required:
        if k not in data:
            raise SystemExit(f"Split is missing {k}; rebuild with pruning_ef1.py")
    src, dst, rid = data["src"].long(), data["dst"].long(), data["rid"].long()
    undirected, node_type = data["undirected"].bool(), data["node_type"].long()
    rel_names, type_names = list(data["rel_names"]), list(data["type_names"])
    n_nodes, n_rels = int(data["num_nodes"]), int(data["num_rels"])
    drug_t, dis_t = type_names.index("DRG"), type_names.index("DIS")
    ind_rids = data.get("ind_rids", torch.tensor([i for i, x in enumerate(rel_names) if x.upper().startswith("DRG-DIS|") and "INDICATION" in x.upper()])).long()
    con_rids = data.get("contra_rids", torch.tensor([i for i, x in enumerate(rel_names) if x.upper().startswith("DRG-DIS|") and "CONTRAINDICATION" in x.upper()])).long()
    if not ind_rids.numel() or not con_rids.numel():
        raise SystemExit("Split must contain both DRG-DIS|INDICATION and DRG-DIS|CONTRAINDICATION relations")
    ind_rids, con_rids = ind_rids.to(dev), con_rids.to(dev)
    split = {k: data["split"][k].long() for k in ("train", "val", "test")}

    drug_ids = torch.nonzero(node_type == drug_t).flatten()
    dis_ids = torch.nonzero(node_type == dis_t).flatten()
    if not drug_ids.numel() or not dis_ids.numel():
        raise SystemExit("No drug or disease candidates in graph")
    drug_pos = torch.full((n_nodes,), -1, dtype=torch.long)
    dis_pos = torch.full((n_nodes,), -1, dtype=torch.long)
    drug_pos[drug_ids] = torch.arange(drug_ids.numel())
    dis_pos[dis_ids] = torch.arange(dis_ids.numel())

    # Split positives for both relations, with the same held-out pair used for both query directions.
    tasks = {}
    for rel_name, rids in (("INDICATION", ind_rids.cpu()), ("CONTRAINDICATION", con_rids.cpu())):
        tasks[rel_name] = {}
        for direction in ("drug_to_disease", "disease_to_drug"):
            cand = dis_ids if direction == "drug_to_disease" else drug_ids
            lut = dis_pos if direction == "drug_to_disease" else drug_pos
            tasks[rel_name][direction] = {}
            for part in ("train", "val", "test"):
                dr, ds = oriented_pairs(src, dst, rid, node_type, split[part], rids, drug_t, dis_t)
                q, target = (dr, ds) if direction == "drug_to_disease" else (ds, dr)
                pos = lut[target] if target.numel() else torch.empty(0, dtype=torch.long)
                if pos.numel() and (pos < 0).any():
                    raise SystemExit(f"{rel_name} {direction}: positive target not in candidate set")
                tasks[rel_name][direction][part] = (q, pos)

    # All known positives are filtered for evaluation/training negatives, per relation and direction.
    known = {}
    for rel_name in tasks:
        known[rel_name] = {}
        for direction in tasks[rel_name]:
            qs, ps = [], []
            for part in ("train", "val", "test"):
                q, p = tasks[rel_name][direction][part]
                qs.append(q); ps.append(p)
            known[rel_name][direction] = (torch.cat(qs).to(dev), torch.cat(ps).to(dev))

    train_idx = split["train"]
    blocked = torch.tensor([i for i, name in enumerate(rel_names) if name.upper().startswith("DRG-DIS|")], dtype=torch.long)
    mp_s, mp_d, mp_r = build_message_graph(src, dst, rid, undirected, train_idx, blocked, n_rels)
    if not mp_s.numel():
        raise SystemExit("Message-passing graph is empty")
    edge_index = torch.stack([mp_s, mp_d]).to(dev)
    edge_type = mp_r.to(dev)
    node_type_d = node_type.to(dev)

    model = RGCN(n_nodes, n_rels, len(type_names), args.dim, args.layers, args.dropout, args.bases).to(dev)
    conv_params = list(model.convs.parameters()) + list(model.norms.parameters())
    conv_ids = {id(p) for p in conv_params}
    other_params = [p for p in model.parameters() if id(p) not in conv_ids]
    opt = torch.optim.Adam([{"params": conv_params, "lr": args.lr * 0.1}, {"params": other_params, "lr": args.lr}])

    def encode():
        return model.encode(edge_index, edge_type, node_type_d, args.edge_drop if model.training else 0.0)

    def evaluate(part):
        model.eval(); hidden = encode(); out = {}
        with torch.no_grad():
            for rel_name, rids in (("INDICATION", ind_rids), ("CONTRAINDICATION", con_rids)):
                # If multiple relation IDs encode synonyms, score each using its own relation vector and aggregate max.
                # Current OptimusKG exports normally have one ID per semantic relation.
                rid0 = int(rids[0])
                out[rel_name] = {}
                for direction in ("drug_to_disease", "disease_to_drug"):
                    q, p = tasks[rel_name][direction][part]
                    cand = dis_ids if direction == "drug_to_disease" else drug_ids
                    kq, kp = known[rel_name][direction]
                    out[rel_name][direction] = metric_block(model, hidden, q.to(dev), p.to(dev), cand.to(dev), kq, kp, rid0)
        model.train()
        return out

    def validation_score(metrics):
        vals = [metrics[r][d]["AUPRC"] for r in metrics for d in metrics[r] if metrics[r][d]["n"]]
        vals = [x for x in vals if math.isfinite(x)]
        return sum(vals) / len(vals) if vals else float("nan")

    def sample_task(rel_name, direction, batch_size):
        q, p = tasks[rel_name][direction]["train"]
        if q.numel() == 0:
            return None
        ix = torch.randint(q.numel(), (batch_size,))
        return q[ix].to(dev), p[ix].to(dev)

    positive_counts = [tasks[r][d]["train"][0].numel() for r in tasks for d in tasks[r]]
    if not any(positive_counts):
        raise SystemExit("No training positives found")
    bs = args.bs
    best = {"score": -1.0, "step": 0, "state": None, "metrics": None}
    stall, curve = 0, []
    t0 = time.time()
    for step in range(0, args.steps + 1):
        if step == 0 or step % args.eval_every == 0 or step == args.steps:
            val = evaluate("val")
            score = validation_score(val)
            curve.append({"step": step, "metrics": val, "mean_AUPRC": score})
            if math.isfinite(score) and score > best["score"] + 1e-5:
                best.update(score=score, step=step, state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, metrics=val)
                stall = 0
            elif step:
                stall += 1
            print(f"[val] step {step} mean AUPRC {score:.4f}", flush=True)
            for rel in val:
                for direction, block in val[rel].items():
                    print(f"  {rel} {direction}: {fmt(block)}", flush=True)
            if step and stall >= args.patience:
                print("Early stopping: validation mean AUPRC did not improve", flush=True)
                break
        if step == args.steps:
            break
        model.train(); opt.zero_grad(set_to_none=True); hidden = encode()
        losses = []
        for rel_name, rids in (("INDICATION", ind_rids), ("CONTRAINDICATION", con_rids)):
            rid0 = int(rids[0])
            for direction in ("drug_to_disease", "disease_to_drug"):
                sample = sample_task(rel_name, direction, bs)
                if sample is None:
                    continue
                q, pos = sample
                cand = dis_ids if direction == "drug_to_disease" else drug_ids
                scores = relation_score(model, hidden, q, cand.to(dev), rid0)
                known_q, known_p = known[rel_name][direction]
                mask = torch.zeros_like(scores, dtype=torch.bool)
                # Mask all other known true tails for this query; preserve this row's target.
                for row, query in enumerate(q.tolist()):
                    positives = known_p[known_q == query]
                    if positives.numel():
                        mask[row, positives] = True
                    mask[row, pos[row]] = False
                losses.append(indication_loss(scores, pos, mask, args.alpha, args.gamma, args.uniform))
        if not losses:
            raise SystemExit("No relation-direction batches available")
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped", flush=True); continue
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 50 == 0:
            elapsed = (time.time() - t0) / max(step, 1)
            print(f"step {step} loss {loss.item():.4f} {elapsed:.3f}s/step", flush=True)

    if best["state"] is None:
        raise SystemExit("No valid checkpoint selected; check that both relations have validation positives")
    model.load_state_dict(best["state"])
    result = {"tag": args.tag, "split": "90/5/5", "best_step": best["step"], "val": best["metrics"],
              "val_curve": curve, "graph": {"nodes": n_nodes, "message_passing_edges": int(mp_s.numel()),
              "drugs": int(drug_ids.numel()), "diseases": int(dis_ids.numel())},
              "args": vars(args)}
    if args.eval_test:
        result["test"] = evaluate("test")
        for rel in result["test"]:
            for direction, block in result["test"][rel].items():
                print(f"[test] {rel} {direction}: {fmt(block)}", flush=True)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "dim": args.dim, "layers": args.layers, "bases": args.bases,
                "ind_rids": ind_rids.cpu(), "contra_rids": con_rids.cpu(), "best_step": best["step"]}, out / f"rgcn_{args.tag}.pt")
    with open(out / f"res_{args.tag}.json", "w") as f:
        json.dump(result, f, indent=2, allow_nan=True)
    print(f"Saved {out / f'res_{args.tag}.json'}", flush=True)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="R-GCN indication/contraindication ranking in both directions")
    p.add_argument("--split", default="artifacts/pruned/okg_split.pt")
    p.add_argument("--out_dir", default="artifacts/pruned")
    p.add_argument("--tag", default="pruned_bidir")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--bases", type=int, default=0)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--edge_drop", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
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
