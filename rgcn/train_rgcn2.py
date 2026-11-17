"""Train a ranker for held-out drug–disease indications.

prep.py reads nodes.parquet and edges.parquet and writes the split.
Message features are every training edge except DRG-DIS, so the model has to
recover a held-out indication from shared genes, pathways, and phenotypes.
Validation reports AUPRC, MRR, and Hits on that ranking. AUPRC is the average
over drugs. The saved checkpoint is the best validation AUPRC, including the
structural score before any training.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from model import (
    PathRanker,
    average_precision,
    build_overlap_channels,
    pair_mask,
    ranks_from_scores,
    summarize_ranks,
)


def device_of(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def orient_indications(src, dst, rid, node_type, edge_ids, ind_rid, drug_t, dis_t):
    empty = torch.empty(0, dtype=torch.long)
    if edge_ids.numel() == 0:
        return empty, empty
    s, t, r = src[edge_ids], dst[edge_ids], rid[edge_ids]
    take = r == ind_rid
    s, t = s[take], t[take]
    if s.numel() == 0:
        return empty, empty
    drug_src = (node_type[s] == drug_t) & (node_type[t] == dis_t)
    drug_dst = (node_type[t] == drug_t) & (node_type[s] == dis_t)
    ok = drug_src | drug_dst
    drug = torch.where(drug_src[ok], s[ok], t[ok])
    dis = torch.where(drug_src[ok], t[ok], s[ok])
    return drug.long(), dis.long()


def pack_examples(drug, dis, dis_ids, is_neuro):
    if drug.numel() == 0:
        return drug, drug, drug.bool()
    pos = torch.searchsorted(dis_ids, dis)
    if not torch.equal(dis_ids[pos], dis):
        raise SystemExit("an indication disease is missing from the candidate list")
    return drug.long(), pos.long(), is_neuro[dis].bool()


def group_positions(drug, pos):
    grouped = defaultdict(list)
    for d, p in zip(drug.tolist(), pos.tolist()):
        grouped[int(d)].append(int(p))
    return grouped


@torch.no_grad()
def evaluate(model, drug, pos, neuro, dis_ids, all_known, split_pos, other_pos):
    """Filtered MRR / Hits and per-drug AUPRC. Candidates are the diseases in the graph."""
    nan = {"n": 0, "AUPRC": float("nan"), "MRR": float("nan"), "H1": float("nan"),
           "H3": float("nan"), "H10": float("nan")}
    if drug.numel() == 0:
        return {"INDICATION": nan}
    scores = model.score_candidates(drug, dis_ids)
    known = pair_mask(drug, pos, all_known[0], all_known[1], dis_ids.numel())
    ranks = ranks_from_scores(scores, pos, known)

    drugs_sorted, order = torch.sort(drug)
    pos_sorted = pos[order]
    first = torch.ones(drugs_sorted.shape[0], dtype=torch.bool, device=drug.device)
    first[1:] = drugs_sorted[1:] != drugs_sorted[:-1]
    uniq_drugs = drugs_sorted[first]
    uniq_scores = model.score_candidates(uniq_drugs, dis_ids)
    # Map each edge onto the unique-drug row so the per-drug label list is complete.
    del pos_sorted

    def pooled(positive_for):
        labels, chosen, macros = [], [], []
        for row, d in enumerate(uniq_drugs.tolist()):
            mine = positive_for.get(int(d), [])
            if not mine:
                continue
            y = torch.zeros(dis_ids.numel(), device=drug.device)
            y[mine] = 1
            keep = torch.ones(dis_ids.numel(), dtype=torch.bool, device=drug.device)
            others = other_pos.get(int(d), [])
            if others:
                keep[others] = False
            keep[mine] = True
            labels.append(y[keep])
            chosen.append(uniq_scores[row][keep])
            macros.append(average_precision(y[keep], uniq_scores[row][keep]))
        if not labels:
            return float("nan"), float("nan")
        micro = average_precision(torch.cat(labels), torch.cat(chosen))
        macro = float(sum(macros) / len(macros))
        return micro, macro

    out = {}
    micro, macro = pooled(split_pos)
    block = summarize_ranks(ranks)
    block["AUPRC"] = round(macro, 4)
    block["AUPRC_micro"] = round(micro, 4)
    out["INDICATION"] = block
    if int(neuro.sum()) and int(neuro.sum()) < int(neuro.numel()):
        neuro_pos = defaultdict(list)
        for d, p, flag in zip(drug.tolist(), pos.tolist(), neuro.tolist()):
            if flag:
                neuro_pos[int(d)].append(int(p))
        micro_n, macro_n = pooled(neuro_pos)
        sub = summarize_ranks(ranks[neuro])
        sub["AUPRC"] = round(macro_n, 4)
        sub["AUPRC_micro"] = round(micro_n, 4)
        out["INDICATION_neuro"] = sub
    return out


def fmt_metrics(block):
    if not block or not block.get("n"):
        return "n=0"
    micro = block.get("AUPRC_micro")
    micro_s = f" micro={micro:.4f}" if isinstance(micro, float) else ""
    return (f"n={block['n']} AUPRC={block['AUPRC']:.4f}{micro_s} MRR={block['MRR']:.4f} "
            f"H1={block['H1']:.4f} H3={block['H3']:.4f} H10={block['H10']:.4f}")


def shared_feature_count(model, drug, pos):
    """How many held-out indications have any path to the disease left in the graph."""
    if drug.numel() == 0:
        return 0
    total = 0
    for start in range(0, drug.numel(), 256):
        sl = slice(start, start + 256)
        scores = model.score_candidates(drug[sl], model.dis_ids)
        rows = torch.arange(scores.shape[0], device=scores.device)
        total += int((scores[rows, pos[sl]] > 0).sum())
    return total


def train(args):
    dev = device_of(args.device)
    torch.manual_seed(args.seed)
    data = torch.load(args.split, map_location="cpu", weights_only=False)
    for key in ("src", "dst", "rid", "undirected", "node_type", "split", "type_names", "rel_names", "ind_rid"):
        if key not in data:
            raise SystemExit(f"{args.split} has no '{key}'. Rebuild it with prep.py.")

    src = data["src"].long()
    dst = data["dst"].long()
    rid = data["rid"].long()
    node_type = data["node_type"].long()
    type_names = list(data["type_names"])
    rel_names = list(data["rel_names"])
    n_nodes = int(data["num_nodes"])
    try:
        drug_t = type_names.index("DRG")
        dis_t = type_names.index("DIS")
    except ValueError as exc:
        raise SystemExit(f"split is missing DRG or DIS: {type_names}") from exc
    ind_rid = int(data["ind_rid"])
    is_neuro = data.get("is_neuro", torch.ones(n_nodes, dtype=torch.bool)).bool()

    keep = torch.ones(n_nodes, dtype=torch.bool)
    if args.keep_nodes:
        import polars as pl
        wanted = [str(i).strip() for i in pl.read_parquet(args.keep_nodes).select("id").unique()["id"].to_list()]
        lookup = {}
        for nid, raw in enumerate(data.get("node_ids", [])):
            lookup[str(raw)] = nid
            if isinstance(raw, str) and ":" in raw:
                lookup.setdefault(raw.replace(":", "_"), nid)
            if isinstance(raw, str) and "_" in raw:
                lookup.setdefault(raw.replace("_", ":"), nid)
        keep = torch.zeros(n_nodes, dtype=torch.bool)
        for raw in wanted:
            nid = lookup.get(raw)
            if nid is None and ":" in raw:
                nid = lookup.get(raw.replace(":", "_"))
            if nid is None and "_" in raw:
                nid = lookup.get(raw.replace("_", ":"))
            if nid is not None:
                keep[nid] = True
        if int(keep.sum()) == 0:
            raise SystemExit(f"none of the ids in {args.keep_nodes} are in this split")
        print(f"keep_nodes {int(keep.sum())}/{n_nodes}", flush=True)

    def subset(edge_ids):
        edge_ids = edge_ids.long()
        return edge_ids[keep[src[edge_ids]] & keep[dst[edge_ids]]]

    blocked = [i for i, name in enumerate(rel_names) if name.startswith("DRG-DIS|")]
    blocked_t = torch.tensor(blocked, dtype=torch.long)
    train_idx = subset(data["split"]["train"])
    if blocked_t.numel() and train_idx.numel():
        pool = train_idx[~torch.isin(rid[train_idx], blocked_t)]
    else:
        pool = train_idx
    print("DRG-DIS edges are held out of the score: "
          + ", ".join(rel_names[i] for i in blocked), flush=True)

    dis_ids = torch.nonzero((node_type == dis_t) & keep).flatten()
    if dis_ids.numel() == 0:
        raise SystemExit("no disease nodes left to rank")

    per_split = {}
    for name in ("train", "val", "test", "test_gen"):
        if name not in data["split"]:
            continue
        drug, dis = orient_indications(
            src, dst, rid, node_type, subset(data["split"][name]), ind_rid, drug_t, dis_t)
        if drug.numel():
            ok = keep[drug] & keep[dis]
            drug, dis = drug[ok], dis[ok]
        per_split[name] = pack_examples(drug, dis, dis_ids, is_neuro)

    tr_drug, tr_pos, _ = per_split["train"]
    if tr_drug.numel() == 0:
        raise SystemExit("no training indications left")
    n_dis = int(dis_ids.numel())
    chance_mrr = (math.log(n_dis) + 0.57721) / n_dis if n_dis > 1 else 1.0
    print(f"indications | train {tr_drug.numel()} val {per_split['val'][0].numel()} "
          f"test {per_split['test'][0].numel()} | candidate diseases {n_dis}", flush=True)
    print(f"random ranking on {n_dis} diseases | MRR ≈ {chance_mrr:.4f} | "
          f"Hits@10 ≈ {min(1.0, 10 / n_dis):.4f}", flush=True)
    print("AUPRC is the average over drugs of how well each held-out disease is ranked", flush=True)
    if data.get("shared_gene_frac") is not None:
        print(f"prep shared-gene rate {data['shared_gene_frac']:.1%}", flush=True)

    channels = build_overlap_channels(
        src, dst, rid, pool, node_type, type_names, rel_names, dis_ids, n_nodes,
    )
    model = PathRanker(channels, dis_ids).to(dev)
    dis_ids = model.dis_ids
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    all_drug, all_pos = [], []
    grouped = {}
    for name, (drug, pos, neuro) in per_split.items():
        grouped[name] = group_positions(drug, pos)
        all_drug.extend(int(d) for d in drug.tolist())
        all_pos.extend(int(p) for p in pos.tolist())
    all_known = (
        torch.tensor(all_drug, dtype=torch.long, device=dev),
        torch.tensor(all_pos, dtype=torch.long, device=dev),
    )
    tr_drug = tr_drug.to(dev)
    tr_pos = tr_pos.to(dev)

    def others_for(name):
        merged = defaultdict(list)
        for other, rows in grouped.items():
            if other == name:
                continue
            for drug_id, positions in rows.items():
                merged[drug_id].extend(positions)
        return merged

    @torch.no_grad()
    def run_eval(name):
        drug, pos, neuro = per_split[name]
        return evaluate(
            model, drug.to(dev), pos.to(dev), neuro.to(dev),
            dis_ids, all_known, grouped[name], others_for(name),
        )

    with torch.no_grad():
        drug, pos, _ = per_split["val"]
        hit = shared_feature_count(model, drug.to(dev), pos.to(dev))
        print(f"val: {hit}/{drug.numel()} indications share a gene, pathway, "
              f"or phenotype with the drug", flush=True)
        if drug.numel() and hit / drug.numel() < 0.3:
            print("warning: most held-out indications have no path to the disease in the "
                  "training graph, so many of them will stay near a random rank", flush=True)

    best = {"score": -1.0, "mrr": -1.0, "step": 0, "state": None, "metrics": None}
    stall = 0
    curve = []

    def consider(step, metrics):
        nonlocal stall
        block = metrics.get("INDICATION", {})
        score = block.get("AUPRC", float("nan"))
        mrr = block.get("MRR", float("nan"))
        better = score == score and (
            best["state"] is None
            or score > best["score"] + 1e-4
            or (score >= best["score"] - 1e-4 and mrr == mrr and mrr > best["mrr"] + 1e-4)
        )
        curve.append({"step": step, "val": metrics})
        if better:
            best.update(score=score, mrr=mrr, step=step, metrics=metrics,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            stall = 0
        else:
            stall += 1

    bs = min(args.bs, int(tr_drug.numel()))
    print(f"device {dev} | channels {', '.join(channel.name for channel in model.channels)}", flush=True)
    t0 = time.time()
    consider(0, run_eval("val"))
    print(f"[val] step 0 {fmt_metrics(best['metrics']['INDICATION'])}", flush=True)

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        choice = torch.randint(0, tr_drug.numel(), (bs,), device=dev)
        scores = model.score_candidates(tr_drug[choice], dis_ids)
        mask = pair_mask(tr_drug[choice], tr_pos[choice], all_known[0], all_known[1], dis_ids.numel())
        # Other known indications of this drug are not negatives.
        scores = scores.masked_fill(mask, -1e4)
        loss = F.cross_entropy(scores * 4.0, tr_pos[choice])
        if not torch.isfinite(loss):
            print(f"step {step} non-finite loss, skipping", flush=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = run_eval("val")
            consider(step, metrics)
            mark = " *" if best["step"] == step else ""
            print(f"[val] step {step} {fmt_metrics(metrics['INDICATION'])}{mark}", flush=True)
            if stall >= args.patience:
                print(f"stopping at step {step}: validation AUPRC did not improve for {args.patience} evals", flush=True)
                break

    if best["state"] is None:
        raise SystemExit("training produced no checkpoint")
    model.load_state_dict(best["state"])
    result = {
        "tag": args.tag,
        "best_step": best["step"],
        "val": best["metrics"],
        "shared_gene_frac": data.get("shared_gene_frac"),
        "channels": [channel.name for channel in model.channels],
        "graph": {
            "nodes": n_nodes,
            "train_edges": int(pool.numel()),
            "train_indications": int(tr_drug.numel()),
            "diseases": int(dis_ids.numel()),
        },
        "val_curve": curve,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    if args.eval_test:
        result["test"] = run_eval("test")
        print(f"[test] step {best['step']} {fmt_metrics(result['test']['INDICATION'])}", flush=True)
        if per_split.get("test_gen", (torch.empty(0),))[0].numel():
            result["test_gen"] = run_eval("test_gen")
            print(f"[test_gen] {fmt_metrics(result['test_gen'].get('INDICATION', {}))}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best["state"], "ind_rid": ind_rid, "best_step": best["step"],
                "channels": result["channels"]},
               out_dir / f"rgcn_{args.tag}.pt")
    with open(out_dir / f"res_{args.tag}.json", "w") as handle:
        json.dump(result, handle, indent=1)
    print(f"saved {out_dir / f'rgcn_{args.tag}.pt'} | best step {best['step']} "
          f"{fmt_metrics(best['metrics']['INDICATION'])}", flush=True)
    return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Rank held-out drug–disease indications.")
    p.add_argument("--split", default="artifacts/okg_split.pt")
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--tag", default="pruned")
    p.add_argument("--keep_nodes", default=None, help="optional extra node list (id column) already in the split")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--bases", type=int, default=0, help="accepted for older job scripts")
    p.add_argument("--layers", type=int, default=2, help="accepted for older job scripts")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--edge_drop", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--aux", type=float, default=0.0, help="accepted for older job scripts")
    p.add_argument("--neg", type=int, default=64)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--uniform", type=float, default=0.25)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--patience", type=int, default=10, help="evals without AUPRC improvement before stopping")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval_test", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--evalset", default=None, help=argparse.SUPPRESS)
    p.add_argument("--fan", nargs="*", default=None, help=argparse.SUPPRESS)
    p.add_argument("--mp_frac", type=float, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not Path(args.split).exists():
        raise SystemExit(f"missing {args.split}. Run: python rgcn/prep.py --data_dir <dir with nodes.parquet>")
    train(args)


if __name__ == "__main__":
    main()
