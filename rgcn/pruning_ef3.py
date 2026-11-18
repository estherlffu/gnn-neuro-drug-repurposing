"""Build the OptimusKG graph and a pair-level indication-only 90/5/5 split.

Uses the 79-disease seed set in seed_diseases_neurodeg.csv. Contraindication
edges are retained in the graph artifact for bookkeeping, but they are not
split into prediction targets and are never used by the indication trainer.
"""
import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okg_files import find_table

ALLOW = {
    "DRG-DIS", "DRG-GEN", "DIS-GEN", "GEN-GEN",
    "DIS-DIS", "DIS-PHE", "DRG-PHE", "PHE-GEN",
    "BPO-GEN", "PWY-GEN", "BPO-BPO", "PWY-PWY", "PHE-PHE",
}
# Large labels are only useful when they touch a seed. This drops the long
# tail of DIS-GEN (about 9.7M edges) that never reaches the diseases we score.
ANCHOR_LABELS = {
    "DIS-GEN", "DIS-PHE", "PHE-GEN", "BPO-GEN", "PWY-GEN",
    "BPO-BPO", "PWY-PWY", "PHE-PHE", "DIS-DIS",
}


def seed_dir_default():
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "data" / "seeds",
        here.parent / "pruning" / "seeds",
        here / "seeds",
        Path("data/seeds"),
        Path("pruning/seeds"),
        Path("keep"),
    ]
    for path in candidates:
        if (path / "seed_diseases_neurodeg.csv").exists() or (path / "seed_nodes_core.csv").exists():
            return path
    return candidates[0]


def add_id_variants(lookup, raw_id, nid):
    lookup[raw_id] = nid
    alts = set()
    if ":" in raw_id:
        alts.add(raw_id.replace(":", "_"))
    if "_" in raw_id:
        alts.add(raw_id.replace("_", ":"))
    for alt in alts:
        lookup.setdefault(alt, nid)


def read_ids(path):
    if path is None or not Path(path).exists():
        return []
    frame = pl.read_csv(path)
    if "id" not in frame.columns:
        raise SystemExit(f"{path} has no id column")
    return [str(i).strip() for i in frame["id"].to_list() if i is not None and str(i).strip()]


def topk_per_group(group, neigh, score, k, rng):
    """Up to k neighbors per group id, highest score first. Ties follow `rng`."""
    if len(group) == 0 or k <= 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((rng.random(len(group)), -np.asarray(score, dtype=np.float64), group))
    grouped = group[order]
    chosen = neigh[order]
    change = np.empty(len(grouped), dtype=bool)
    change[0] = True
    if len(grouped) > 1:
        change[1:] = grouped[1:] != grouped[:-1]
    starts = np.maximum.accumulate(np.where(change, np.arange(len(grouped)), 0))
    rank = np.arange(len(grouped)) - starts
    return chosen[rank < k]


def oriented(src, dst, src_t, dst_t, type_a, type_b):
    forward = (src_t == type_a) & (dst_t == type_b)
    backward = (src_t == type_b) & (dst_t == type_a)
    take = forward | backward
    a = np.where(src_t[take] == type_a, src[take], dst[take])
    b = np.where(src_t[take] == type_a, dst[take], src[take])
    return a, b


def cap_undirected(src, dst, k, rng):
    """Keep edges so each endpoint has at most k of these neighbors."""
    if len(src) == 0:
        return np.empty(0, dtype=bool)
    edge_id = np.arange(len(src))
    group = np.concatenate([src, dst])
    neigh = np.concatenate([dst, src])
    eid = np.concatenate([edge_id, edge_id])
    score = np.zeros(len(group))
    picked = topk_per_group(group, eid, score, k, rng)
    keep = np.zeros(len(src), dtype=bool)
    keep[np.unique(picked)] = True
    return keep


def load_edges(edge_path, anchor_ids):
    scan = pl.scan_parquet(edge_path)
    try:
        columns = scan.collect_schema().names()
    except AttributeError:
        columns = scan.schema.names()
    wanted = [c for c in ("from", "to", "label", "relation", "undirected") if c in columns]
    missing = {"from", "to", "label", "relation"} - set(wanted)
    if missing:
        raise SystemExit(f"{edge_path} is missing columns: {sorted(missing)}")
    frame = (
        pl.scan_parquet(edge_path)
        .select(wanted)
        .with_columns(
            pl.col("from").cast(pl.Utf8),
            pl.col("to").cast(pl.Utf8),
            pl.col("label").cast(pl.Utf8),
            pl.col("relation").cast(pl.Utf8),
        )
        .filter(pl.col("label").is_in(sorted(ALLOW)))
        .collect()
    )
    if frame.height == 0:
        labels = pl.scan_parquet(edge_path).select(pl.col("label").unique()).collect()["label"].to_list()
        raise SystemExit(f"no allow-listed edges in {edge_path}. Labels present: {labels[:40]}")
    if "undirected" not in frame.columns:
        frame = frame.with_columns(pl.lit(False).alias("undirected"))
    frame = frame.with_columns(pl.col("undirected").fill_null(False))
    anchored = (
        ~pl.col("label").is_in(sorted(ANCHOR_LABELS))
        | pl.col("from").is_in(anchor_ids)
        | pl.col("to").is_in(anchor_ids)
    )
    before = frame.height
    frame = frame.filter(anchored)
    print(f"allow-listed edges {before:,} | touching a seed (or a small relation) {frame.height:,}", flush=True)
    return frame


def select_nodes(src, dst, src_t, dst_t, labels, codes, seeds, n_nodes, max_genes, term_genes, max_pheno, gene_neighbors, rng):
    """Return a boolean mask over nodes. Diseases in the mask are the seeds only."""
    n = n_nodes
    code = codes
    dis = seeds["dis"]
    gene_seeds = seeds["gene"]
    term_seeds = seeds["term"]

    def edges(label, a, b):
        empty = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if a not in code or b not in code:
            return empty
        take = labels == label
        if not np.any(take):
            return empty
        return oriented(src[take], dst[take], src_t[take], dst_t[take], code[a], code[b])

    dis_node, gene_node = edges("DIS-GEN", "DIS", "GEN")
    seed_dis_edge = np.isin(dis_node, dis) if len(dis_node) else np.empty(0, dtype=bool)
    shared = np.bincount(gene_node[seed_dis_edge], minlength=n) if seed_dis_edge.any() else np.zeros(n, dtype=np.int64)

    term_gene_a, term_gene_b = [], []
    for label, term_type in (("BPO-GEN", "BPO"), ("PWY-GEN", "PWY")):
        a, b = edges(label, term_type, "GEN")
        if len(a):
            term_gene_a.append(a)
            term_gene_b.append(b)
    if term_gene_a:
        term_ids = np.concatenate(term_gene_a)
        term_genes_arr = np.concatenate(term_gene_b)
        linked = np.zeros(n, dtype=bool)
        hit = np.isin(term_ids, term_seeds)
        if hit.any():
            linked[term_genes_arr[hit]] = True
    else:
        term_ids = np.empty(0, dtype=np.int64)
        term_genes_arr = np.empty(0, dtype=np.int64)
        linked = np.zeros(n, dtype=bool)

    is_seed_gene = np.zeros(n, dtype=bool)
    is_seed_gene[gene_seeds] = True
    gene_score_table = shared.astype(np.float64) * 10.0 + linked.astype(np.float64) * 5.0 + is_seed_gene.astype(np.float64) * 1e6

    genes = set(int(g) for g in gene_seeds)
    if len(dis_node):
        keep_g = topk_per_group(dis_node[seed_dis_edge], gene_node[seed_dis_edge], gene_score_table[gene_node[seed_dis_edge]], max_genes, rng)
        genes.update(int(g) for g in keep_g)
    if len(term_ids):
        seed_term_edge = np.isin(term_ids, term_seeds)
        extra = topk_per_group(term_ids[seed_term_edge], term_genes_arr[seed_term_edge], gene_score_table[term_genes_arr[seed_term_edge]], term_genes, rng)
        genes.update(int(g) for g in extra)

    drug_d, dis_d = edges("DRG-DIS", "DRG", "DIS")
    drugs = set()
    if len(drug_d):
        hit = np.isin(dis_d, dis)
        drugs.update(int(d) for d in drug_d[hit])

    drug_g, gene_g = edges("DRG-GEN", "DRG", "GEN")
    if len(drug_g) and drugs:
        indicated = np.isin(drug_g, np.fromiter(drugs, dtype=np.int64, count=len(drugs)))
        genes.update(int(g) for g in gene_g[indicated])
    if len(drug_g) and genes:
        gene_arr = np.fromiter(genes, dtype=np.int64, count=len(genes))
        hit = np.isin(gene_g, gene_arr)
        drugs.update(int(d) for d in drug_g[hit])

    phe_d, phe_p = edges("DIS-PHE", "DIS", "PHE")
    phenos = set()
    if len(phe_d):
        hit = np.isin(phe_d, dis)
        shared_phe = np.bincount(phe_p[hit], minlength=n) if hit.any() else np.zeros(n)
        picked = topk_per_group(phe_d[hit], phe_p[hit], shared_phe[phe_p[hit]], max_pheno, rng)
        phenos.update(int(p) for p in picked)

    terms = set(int(t) for t in term_seeds)
    if len(term_ids) and gene_seeds.size:
        hit = np.isin(term_genes_arr, gene_seeds)
        picked = topk_per_group(term_genes_arr[hit], term_ids[hit], np.ones(int(hit.sum())), 10, rng)
        terms.update(int(t) for t in picked)
    for label, term_type in (("BPO-BPO", "BPO"), ("PWY-PWY", "PWY"), ("PHE-PHE", "PHE")):
        a, b = edges(label, term_type, term_type)
        if not len(a):
            continue
        anchors = term_seeds if term_type != "PHE" else np.fromiter(phenos, dtype=np.int64, count=len(phenos))
        if anchors.size == 0:
            continue
        hit = np.isin(a, anchors)
        picked = topk_per_group(a[hit], b[hit], np.ones(int(hit.sum())), 10, rng)
        if term_type == "PHE":
            phenos.update(int(p) for p in picked)
        else:
            terms.update(int(t) for t in picked)

    # Retain all drug nodes represented in the allow-listed edge table, not only
    # drugs already linked to a selected disease by an indication edge.
    if "DRG" in code:
        all_drugs = np.unique(np.concatenate([src[src_t == code["DRG"]], dst[dst_t == code["DRG"]]]))
        drugs.update(int(d) for d in all_drugs)

    mask = np.zeros(n, dtype=bool)
    for group in (dis, genes, drugs, phenos, terms):
        if len(group) == 0:
            continue
        idx = np.fromiter(group, dtype=np.int64, count=len(group)) if not isinstance(group, np.ndarray) else group
        mask[idx] = True
    return mask, gene_neighbors


def apply_gene_neighbor_cap(src, dst, labels, node_mask, cap, rng):
    keep = node_mask[src] & node_mask[dst]
    gene_edge = keep & (labels == "GEN-GEN")
    if cap <= 0 or not gene_edge.any():
        return keep
    idx = np.nonzero(gene_edge)[0]
    chosen = cap_undirected(src[idx], dst[idx], cap, rng)
    keep[idx] = chosen
    return keep


def split_indications(drug, dis, node_fam, n_fam, val_frac, test_frac, seed):
    """Group split of indication edges by (drug, family).

    `drug` and `dis` are aligned arrays for indication edges only.
    Returns es of the same length: 0 train, 1 val, 2 test. Non-family
    diseases stay 0 so the caller can hold them out separately.
    """
    fam = node_fam[dis]
    es = np.zeros(len(drug), dtype=np.int64)
    cand = np.nonzero(fam >= 0)[0]
    if cand.size == 0:
        return es
    g = torch.Generator().manual_seed(seed)
    de = torch.from_numpy(np.ascontiguousarray(drug[cand], dtype=np.int64))
    fe = torch.from_numpy(np.ascontiguousarray(fam[cand], dtype=np.int64))
    _, inv = torch.unique(de * n_fam + fe, return_inverse=True)
    gsize = torch.bincount(inv)
    perm = torch.randperm(gsize.numel(), generator=g)
    cum = torch.cumsum(gsize[perm], 0).float()
    n_test, n_val = test_frac * float(cand.size), val_frac * float(cand.size)
    gs = torch.zeros(gsize.numel(), dtype=torch.long)
    gs[perm[cum <= n_test]] = 2
    gs[perm[(cum > n_test) & (cum <= n_test + n_val)]] = 1
    es[cand] = gs[inv].numpy()
    if not (es[cand] == 1).any() or not (es[cand] == 2).any():
        print("group split left val or test empty; splitting those edges individually", flush=True)
        order = torch.randperm(cand.size, generator=g).numpy()
        n_te = max(1, int(round(test_frac * cand.size)))
        n_va = max(1, int(round(val_frac * cand.size)))
        if n_te + n_va >= cand.size:
            n_va = max(1, cand.size // 5)
            n_te = max(1, cand.size // 5)
        es[cand] = 0
        es[cand[order[:n_te]]] = 2
        es[cand[order[n_te:n_te + n_va]]] = 1
    return es


def build(args):
    node_path = find_table(args.data_dir, "nodes")
    edge_path = find_table(args.data_dir, "edges")
    print(f"nodes: {node_path}", flush=True)
    print(f"edges: {edge_path}", flush=True)
    schema = pl.scan_parquet(node_path).collect_schema().names()
    node_cols = [c for c in ("id", "label", "properties") if c in schema]
    if "id" not in node_cols or "label" not in node_cols:
        raise SystemExit(f"{node_path} must have id and label columns, found {schema}")
    nodes = pl.read_parquet(node_path, columns=node_cols).with_row_index("nid")
    if nodes["id"].dtype != pl.Utf8:
        nodes = nodes.with_columns(pl.col("id").cast(pl.Utf8))
    if nodes["label"].dtype != pl.Utf8:
        nodes = nodes.with_columns(pl.col("label").cast(pl.Utf8))

    lookup = {}
    for nid, raw in zip(nodes["nid"].to_list(), nodes["id"].to_list()):
        add_id_variants(lookup, raw, nid)

    seed_dir = Path(args.seed_dir)
    neuro_rows = pl.read_csv(seed_dir / "seed_diseases_neurodeg.csv") if (seed_dir / "seed_diseases_neurodeg.csv").exists() else None
    if neuro_rows is not None:
        neuro_rows = neuro_rows.with_columns(pl.col("id").cast(pl.Utf8), pl.col("source").cast(pl.Utf8))
    if neuro_rows is None:
        raise SystemExit(f"missing {seed_dir / 'seed_diseases_neurodeg.csv'}")
    disease_ids = read_ids(seed_dir / "seed_diseases_neurodeg.csv")  # evaluation diseases only
    gene_ids = read_ids(seed_dir / "seed_nodes_core.csv")
    term_ids = read_ids(seed_dir / "seed_terms_core.csv")

    def resolve_many(raw_ids):
        found, missing = [], []
        for raw in raw_ids:
            nid = lookup.get(raw.strip())
            if nid is None:
                missing.append(raw)
            else:
                found.append(nid)
        return np.unique(np.asarray(found, dtype=np.int64)), missing

    dis_nids, dis_missing = resolve_many(disease_ids)
    gene_nids, gene_missing = resolve_many(gene_ids)
    term_nids, term_missing = resolve_many(term_ids)
    print(f"seeds matched | diseases {dis_nids.size}/{len(set(disease_ids))} "
          f"genes {gene_nids.size}/{len(set(gene_ids))} terms {term_nids.size}/{len(set(term_ids))}", flush=True)
    if dis_missing[:8]:
        print(f"  unmatched disease ids (first few): {dis_missing[:8]}", flush=True)
    if dis_nids.size == 0:
        sample = nodes["id"].head(5).to_list()
        raise SystemExit(
            "none of the seed diseases are in the node file. "
            f"Seed ids look like {disease_ids[:3]}; node ids look like {sample}."
        )

    anchor = [nodes["id"][int(i)] for i in list(dis_nids) + list(gene_nids) + list(term_nids)]
    # Also match the colon/underscore variant so the edge filter sees both spellings.
    anchor_ids = []
    for raw in anchor:
        anchor_ids.append(raw)
        if ":" in raw:
            anchor_ids.append(raw.replace(":", "_"))
        if "_" in raw:
            anchor_ids.append(raw.replace("_", ":"))
    edges = load_edges(edge_path, list(set(anchor_ids)))
    print(f"joining {edges.height:,} edges to {nodes.height:,} nodes", flush=True)
    indexed = nodes.select("id", "nid", "label")
    joined = (
        edges.join(indexed.rename({"id": "from", "nid": "src", "label": "src_label"}), on="from", how="inner")
        .join(indexed.rename({"id": "to", "nid": "dst", "label": "dst_label"}), on="to", how="inner")
    )
    print(f"edges with both endpoints in nodes.parquet: {joined.height:,}/{edges.height:,}", flush=True)
    if joined.height == 0:
        raise SystemExit("every allow-listed edge pointed at an id that is not in nodes.parquet")

    type_names = sorted(nodes["label"].unique().to_list())
    type_code = {name: i for i, name in enumerate(type_names)}
    for required in ("DRG", "DIS", "GEN"):
        if required not in type_code:
            raise SystemExit(f"node file has no {required!r} label. Labels: {type_names}")

    src = joined["src"].to_numpy().astype(np.int64)
    dst = joined["dst"].to_numpy().astype(np.int64)
    src_t = np.asarray([type_code[x] for x in joined["src_label"].to_list()], dtype=np.int64)
    dst_t = np.asarray([type_code[x] for x in joined["dst_label"].to_list()], dtype=np.int64)
    labels = np.asarray(joined["label"].to_list())
    relations = np.asarray(joined["relation"].to_list())
    undirected_np = joined["undirected"].to_numpy().astype(bool)
    rtype = np.asarray([f"{a}|{b}" for a, b in zip(labels.tolist(), relations.tolist())])

    rng = np.random.default_rng(args.seed)
    n_nodes_full = nodes.height
    if args.no_prune:
        print("not pruning: keeping every allow-listed edge", flush=True)
        node_mask = np.zeros(n_nodes_full, dtype=bool)
        node_mask[src] = True
        node_mask[dst] = True
        edge_keep = np.ones(len(src), dtype=bool)
    else:
        node_mask, _ = select_nodes(
            src, dst, src_t, dst_t, labels, type_code,
            {"dis": dis_nids, "gene": gene_nids, "term": term_nids},
            n_nodes_full, args.max_genes, args.term_genes, args.max_pheno, args.gene_neighbors, rng,
        )
        # Every matched disease seed stays a candidate even with no edges.
        node_mask[dis_nids] = True
        edge_keep = apply_gene_neighbor_cap(src, dst, labels, node_mask, args.gene_neighbors, rng)
        # Drop edges that the node mask no longer wants (gene-neighbor cap does not add nodes).
        edge_keep &= node_mask[src] & node_mask[dst]

    src, dst = src[edge_keep], dst[edge_keep]
    labels, rtype = labels[edge_keep], rtype[edge_keep]
    undirected_np = undirected_np[edge_keep]
    src_t, dst_t = src_t[edge_keep], dst_t[edge_keep]
    touched = np.zeros(n_nodes_full, dtype=bool)
    touched[src] = True
    touched[dst] = True
    if not args.no_prune:
        touched[dis_nids] = True
    node_mask = touched

    kept_idx = np.nonzero(node_mask)[0]
    n = int(kept_idx.size)
    new_id = np.full(n_nodes_full, -1, dtype=np.int64)
    new_id[kept_idx] = np.arange(kept_idx.size)
    src, dst = new_id[src], new_id[dst]
    kept_nodes = (
        pl.DataFrame({"nid": kept_idx.tolist()}).with_row_index("new_id")
        .join(nodes.select([c for c in ("nid", "id", "label", "properties") if c in nodes.columns]), on="nid", how="left")
        .sort("new_id")
    )
    node_ids = kept_nodes["id"].to_list()
    node_labels = kept_nodes["label"].to_list()
    kept_types = sorted(set(node_labels))
    kept_code = {name: i for i, name in enumerate(kept_types)}
    node_type = np.asarray([kept_code[x] for x in node_labels], dtype=np.int64)

    rel_names = sorted(set(rtype.tolist()))
    rel_code = {name: i for i, name in enumerate(rel_names)}
    rid = np.asarray([rel_code[x] for x in rtype.tolist()], dtype=np.int64)
    # One row per directed triple.
    if len(src) == 0:
        raise SystemExit("pruning removed every edge. Check that the seed ids match nodes.parquet.")
    order = np.lexsort((rid, dst, src))
    src, dst, rid, undirected_np = src[order], dst[order], rid[order], undirected_np[order]
    change = np.empty(len(src), dtype=bool)
    change[0] = True
    if len(src) > 1:
        change[1:] = (src[1:] != src[:-1]) | (dst[1:] != dst[:-1]) | (rid[1:] != rid[:-1])
    src, dst, rid = src[change], dst[change], rid[change]
    undirected_np = undirected_np[change]

    # Identify indication relation IDs. Contraindications are deliberately not
    # a prediction task in this experiment because only six pairs are present.
    ind_names = [
        name for name in rel_names
        if name.upper().startswith("DRG-DIS|")
        and name.upper().split("|")[-1] == "INDICATION"
    ]

    contra_names = [
        name for name in rel_names
        if name.upper().startswith("DRG-DIS|")
        and name.upper().split("|")[-1] == "CONTRAINDICATION"
    ]

    if not ind_names:
        raise SystemExit(f"no DRG-DIS indication relation found. Relations: {rel_names}")
        
    ind_rids = np.asarray([rel_code[x] for x in ind_names], dtype=np.int64)
    contra_rids = np.asarray([rel_code[x] for x in contra_names], dtype=np.int64)

    drug_t, dis_t = kept_code["DRG"], kept_code["DIS"]
    st, dt = node_type[src], node_type[dst]
    is_drug_dis = ((st == drug_t) & (dt == dis_t)) | ((st == dis_t) & (dt == drug_t))
    is_ind = np.isin(rid, ind_rids) & is_drug_dis

    def orient_mask(mask):
        ii = np.nonzero(mask)[0]
        drug = np.where(st[ii] == drug_t, src[ii], dst[ii])
        dis = np.where(st[ii] == drug_t, dst[ii], src[ii])
        return ii, drug.astype(np.int64), dis.astype(np.int64)

    def split_pairs(edge_idx, drug, dis, seed):
        """Random 90/5/5 split by unique semantic drug-disease pair."""
        result = np.zeros(len(src), dtype=np.int64)
        pair = drug.astype(np.int64) * n + dis.astype(np.int64)
        if len(np.unique(pair)) != len(pair):
            _, first = np.unique(pair, return_index=True)
            edge_idx, drug, dis, pair = edge_idx[first], drug[first], dis[first], pair[first]
        if len(edge_idx) < 3:
            raise SystemExit(f"Need at least 3 unique indication pairs; found {len(edge_idx)}")
        rng_local = np.random.default_rng(seed)
        order = rng_local.permutation(len(edge_idx))
        n_val = max(1, int(round(0.05 * len(edge_idx))))
        n_test = max(1, int(round(0.05 * len(edge_idx))))
        if n_val + n_test >= len(edge_idx):
            n_val = n_test = 1
        val_local = order[:n_val]
        test_local = order[n_val:n_val + n_test]
        train_local = order[n_val + n_test:]
        result[edge_idx[val_local]] = 1
        result[edge_idx[test_local]] = 2
        return result, edge_idx[train_local], edge_idx[val_local], edge_idx[test_local]

    ind_idx, ind_drug, ind_dis = orient_mask(is_ind)
    es_ind, ind_train, ind_val, ind_test = split_pairs(ind_idx, ind_drug, ind_dis, args.seed + 11)
    val_e = ind_val
    test_e = ind_test
    held = np.concatenate([val_e, test_e])

    # Exclude held-out indication edges and any reverse-oriented copy of the
    # same semantic pair from the training edge pool.
    held_keys = set()
    for e in held:
        held_keys.add((int(src[e]), int(dst[e])))
        held_keys.add((int(dst[e]), int(src[e])))
    drop = np.fromiter(((int(a), int(b)) in held_keys for a, b in zip(src, dst)), dtype=bool, count=len(src))
    train = np.nonzero(~drop)[0]
    print(f"indication pairs train/val/test {len(ind_train)}/{len(ind_val)}/{len(ind_test)}", flush=True)
    print(f"contraindication pairs retained as non-target context: {int((np.isin(rid, contra_rids) & is_drug_dis).sum())}", flush=True)

    # Shared-gene diagnostic for held-out indication pairs.
    gene_t = kept_code["GEN"]
    drug_genes, dis_genes = defaultdict(set), defaultdict(set)
    for s, d, name in zip(src.tolist(), dst.tolist(), (rel_names[i] for i in rid.tolist())):
        if name.startswith("DRG-GEN"):
            a, b = (s, d) if node_type[s] == drug_t and node_type[d] == gene_t else (d, s)
            if node_type[a] == drug_t and node_type[b] == gene_t:
                drug_genes[a].add(b)
        elif name.startswith("DIS-GEN"):
            a, b = (s, d) if node_type[s] == dis_t and node_type[d] == gene_t else (d, s)
            if node_type[a] == dis_t and node_type[b] == gene_t:
                dis_genes[a].add(b)
    shared_hit = sum(bool(drug_genes[int(dr)] & dis_genes[int(di)]) for dr, di in zip(ind_drug.tolist(), ind_dis.tolist()))
    frac = shared_hit / max(1, len(ind_drug))
    print(f"indication pairs whose drug and disease share a gene: {shared_hit}/{len(ind_drug)} ({frac:.1%})", flush=True)

    by_type = Counter(node_labels)
    by_label = Counter(labels.tolist())
    print(f"pruned graph | nodes {n:,} {dict(by_type)}", flush=True)
    print(f"pruned graph | edges {len(src):,} {dict(by_label)}", flush=True)
    if val_e.size == 0 or test_e.size == 0:
        raise SystemExit("validation or test split is empty")

    is_neuro = np.zeros(n, dtype=bool)
    if "properties" in kept_nodes.columns:
        try:
            prop = kept_nodes["properties"].fill_null("").cast(pl.Utf8)
            is_neuro = prop.str.contains("EFO_0000618|EFO:0000618").fill_null(False).to_numpy()
        except Exception as exc:
            print(f"could not read the nervous-system tag from properties ({exc})", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = pl.DataFrame([
        {"relation": "INDICATION", "train": int(len(ind_train)), "val": int(len(ind_val)), "test": int(len(ind_test))},
    ])
    print(report, flush=True)
    report.write_csv(out_dir / "split_report.csv")

    payload = {
        "src": torch.from_numpy(src.astype(np.int64)),
        "dst": torch.from_numpy(dst.astype(np.int64)),
        "rid": torch.from_numpy(rid.astype(np.int64)),
        "undirected": torch.from_numpy(undirected_np.astype(bool)),
        "node_type": torch.from_numpy(node_type.astype(np.int64)),
        "type_names": kept_types,
        "rel_names": rel_names,
        "node_ids": node_ids,
        "split": {
            "train": torch.from_numpy(train.astype(np.int64)),
            "val": torch.from_numpy(val_e.astype(np.int64)),
            "test": torch.from_numpy(test_e.astype(np.int64)),
        },
        "neuro_fam": torch.full((n,), -1, dtype=torch.long),
        "is_neuro": torch.from_numpy(is_neuro.astype(bool)),
        "num_nodes": int(n),
        "num_rels": len(rel_names),
        "num_ntypes": len(kept_types),
        "ind_rid": int(ind_rids[0]),
        "ind_rids": torch.from_numpy(ind_rids),
        "contra_rids": torch.from_numpy(contra_rids),
        "pruned": not args.no_prune,
        "shared_gene_frac": float(frac),
    }
    out_pt = out_dir / "okg_split.pt"
    torch.save(payload, out_pt)
    pl.DataFrame({"id": node_ids, "nid": list(range(n)), "label": node_labels}).write_parquet(out_dir / "node_index.parquet")
    pl.DataFrame({"rtype": rel_names, "rid": list(range(len(rel_names)))}).write_parquet(out_dir / "rel_index.parquet")
    print(f"saved {out_pt} | {n:,} nodes, {len(rel_names)} relations", flush=True)
    return payload


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prune OptimusKG and split indication drug-disease pairs 90/5/5.")
    p.add_argument("--data_dir", default=".", help="directory containing nodes.parquet and edges.parquet")
    p.add_argument("--seed_dir", default=str(seed_dir_default()))
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--max_genes", type=int, default=400, help="genes kept per seed disease, shared and seeded genes first")
    p.add_argument("--term_genes", type=int, default=150, help="genes kept per seed pathway or process")
    p.add_argument("--max_pheno", type=int, default=40, help="phenotypes kept per seed disease")
    p.add_argument("--gene_neighbors", type=int, default=40, help="gene-gene neighbors kept per gene")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_prune", action="store_true", help="keep every allow-listed edge, not only the seed neighborhood")
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_pt = Path(args.out_dir) / "okg_split.pt"
    if out_pt.exists() and not args.force:
        raise SystemExit(f"{out_pt} already exists. Pass --force to rebuild it.")
    build(args)


if __name__ == "__main__":
    main()
