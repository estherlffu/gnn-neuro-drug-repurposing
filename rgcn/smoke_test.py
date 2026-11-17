"""Prove the trainer can rank a drug–disease link that is only visible through a gene.

The graph is synthetic and small. Each drug treats the disease that shares its
target gene. Indication edges are held out of message passing, which is the
same rule as the real run. AUPRC, MRR and Hits have to land well above chance.
"""
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import average_precision, build_message_graph
from train_rgcn2 import parse_args, train


def check_average_precision():
    labels = torch.tensor([1.0, 0.0, 0.0])
    scores = torch.tensor([0.2, 0.9, 0.1])
    # Ranked second of three. One positive => AP = 1/rank = 0.5.
    got = average_precision(labels, scores)
    if abs(got - 0.5) > 1e-5:
        raise SystemExit(f"average_precision returned {got}, expected 0.5")


def synthetic_split(path):
    n_dis, n_drug = 40, 80
    drugs = torch.arange(n_drug)
    genes = torch.arange(n_drug, n_drug + n_dis)
    diseases = torch.arange(n_drug + n_dis, n_drug + 2 * n_dis)
    n = n_drug + 2 * n_dis
    # drug i -> its gene, disease i%40 -- that same gene, drug i indicated for that disease
    gene_of = genes[drugs % n_dis]
    dis_of = diseases[drugs % n_dis]
    src = torch.cat([drugs, diseases, drugs])
    dst = torch.cat([gene_of, genes, dis_of])
    rid = torch.cat([
        torch.zeros(n_drug, dtype=torch.long),
        torch.ones(n_dis, dtype=torch.long),
        torch.full((n_drug,), 2, dtype=torch.long),
    ])
    undirected = torch.cat([
        torch.zeros(n_drug, dtype=torch.bool),
        torch.ones(n_dis, dtype=torch.bool),
        torch.ones(n_drug, dtype=torch.bool),
    ])
    ind = torch.arange(n_drug + n_dis, n_drug + n_dis + n_drug)
    g = torch.Generator().manual_seed(0)
    perm = ind[torch.randperm(ind.numel(), generator=g)]
    n_test, n_val = 16, 16
    test, val = perm[:n_test], perm[n_test:n_test + n_val]
    held = torch.cat([test, val])
    train = torch.ones(src.numel(), dtype=torch.bool)
    train[held] = False
    node_type = torch.cat([
        torch.zeros(n_drug, dtype=torch.long),
        torch.ones(n_dis, dtype=torch.long),
        torch.full((n_dis,), 2, dtype=torch.long),
    ])
    is_neuro = node_type == 2
    blocked = torch.tensor([2])
    mp_s, mp_d, mp_r = build_message_graph(src, dst, rid, undirected, train.nonzero().flatten(), blocked, 3)
    if (mp_r == 2).any() or (mp_r == 2 + 3).any():
        raise SystemExit("indication edges leaked into the message-passing graph")
    payload = {
        "src": src, "dst": dst, "rid": rid, "undirected": undirected,
        "node_type": node_type,
        "type_names": ["DRG", "GEN", "DIS"],
        "rel_names": ["DRG-GEN|TARGET", "DIS-GEN|ASSOCIATED_WITH", "DRG-DIS|INDICATION"],
        "node_ids": [f"n{i}" for i in range(n)],
        "split": {
            "train": train.nonzero().flatten(),
            "val": val,
            "test": test,
            "test_gen": torch.empty(0, dtype=torch.long),
        },
        "neuro_fam": torch.where(is_neuro, torch.zeros(n, dtype=torch.long), torch.full((n,), -1)),
        "is_neuro": is_neuro,
        "num_nodes": n, "num_rels": 3, "num_ntypes": 3, "ind_rid": 2,
        "pruned": True, "shared_gene_frac": 1.0,
    }
    torch.save(payload, path)
    return payload


def check_trainer(path, out_dir):
    args = parse_args([
        "--split", str(path),
        "--out_dir", str(out_dir),
        "--tag", "smoke",
        "--steps", "250",
        "--eval_every", "50",
        "--patience", "20",
        "--bs", "64",
        "--dim", "64",
        "--layers", "2",
        "--dropout", "0",
        "--edge_drop", "0",
        "--lr", "3e-3",
        "--aux", "0.1",
        "--device", "cpu",
        "--eval_test",
        "--seed", "0",
    ])
    result = train(args)
    val = result["val"]["INDICATION"]
    test = result["test"]["INDICATION"]
    print("smoke val", val)
    print("smoke test", test)
    # Chance on 40 diseases is about AUPRC 0.08, MRR 0.10, Hits@10 0.25.
    for name, block in (("val", val), ("test", test)):
        if block["AUPRC"] < 0.5 or block["MRR"] < 0.45 or block["H10"] < 0.75 or block["H1"] < 0.3:
            raise SystemExit(f"{name} metrics are still near chance: {block}")
    worst = min(point["val"]["INDICATION"]["AUPRC"] for point in result["val_curve"])
    if worst < 0.85:
        raise SystemExit(f"training dragged AUPRC down to {worst}")


def check_prep(root):
    import polars as pl
    from prep import build
    data = root / "data"
    seeds = root / "seeds"
    out = root / "out"
    data.mkdir(parents=True)
    seeds.mkdir(parents=True)
    n_dis, n_drug = 30, 60
    ids, labels, props = [], [], []
    for i in range(n_drug):
        ids.append(f"DRUG_{i}")
        labels.append("DRG")
        props.append("{}")
    for i in range(n_dis):
        ids.append(f"GEN_{i}")
        labels.append("GEN")
        props.append("{}")
    for i in range(n_dis):
        ids.append(f"DIS_{i}")
        labels.append("DIS")
        props.append('{"therapeutic_areas": ["EFO_0000618"]}')
    ids += ["DIS_EXTRA", "ANA_0"]
    labels += ["DIS", "ANA"]
    props += ["{}", "{}"]
    pl.DataFrame({"id": ids, "label": labels, "properties": props}).write_parquet(data / "nodes.parquet")

    src, dst, label, relation, und = [], [], [], [], []
    for i in range(n_drug):
        src.append(f"DRUG_{i}")
        dst.append(f"GEN_{i % n_dis}")
        label.append("DRG-GEN")
        relation.append("TARGET")
        und.append(False)
        src.append(f"DRUG_{i}")
        dst.append(f"DIS_{i % n_dis}")
        label.append("DRG-DIS")
        relation.append("INDICATION")
        und.append(True)
    for i in range(n_dis):
        src.append(f"DIS_{i}")
        dst.append(f"GEN_{i}")
        label.append("DIS-GEN")
        relation.append("ASSOCIATED_WITH")
        und.append(True)
    src += ["DIS_EXTRA", "ANA_0"]
    dst += ["GEN_0", "GEN_0"]
    label += ["DIS-GEN", "ANA-GEN"]
    relation += ["ASSOCIATED_WITH", "EXPRESSION_PRESENT"]
    und += [True, False]
    # A repurposing candidate: targets a kept gene, no indication.
    src.append("DRUG_EXTRA")
    dst.append("GEN_0")
    label.append("DRG-GEN")
    relation.append("TARGET")
    und.append(False)
    ids.append("DRUG_EXTRA")  # not in the node frame yet — add it
    # DRUG_EXTRA was not written. Rewrite nodes with it included.
    ids2 = ids  # includes DRUG_EXTRA at the end but nodes were already written without it
    # Rebuild nodes so DRUG_EXTRA exists and DIS_EXTRA / ANA stay distractors.
    node_ids = [f"DRUG_{i}" for i in range(n_drug)] + ["DRUG_EXTRA"]
    node_labels = ["DRG"] * n_drug + ["DRG"]
    node_props = ["{}"] * (n_drug + 1)
    node_ids += [f"GEN_{i}" for i in range(n_dis)]
    node_labels += ["GEN"] * n_dis
    node_props += ["{}"] * n_dis
    node_ids += [f"DIS_{i}" for i in range(n_dis)] + ["DIS_EXTRA", "ANA_0"]
    node_labels += ["DIS"] * n_dis + ["DIS", "ANA"]
    node_props += ['{"therapeutic_areas": ["EFO_0000618"]}'] * n_dis + ["{}", "{}"]
    pl.DataFrame({"id": node_ids, "label": node_labels, "properties": node_props}).write_parquet(data / "nodes.parquet")
    pl.DataFrame({
        "from": src, "to": dst, "label": label, "relation": relation, "undirected": und,
    }).write_parquet(data / "edges.parquet")

    rows = ["id,name,source"]
    rows.append("DIS_0,d0,root")
    for i in range(1, n_dis // 2):
        rows.append(f"DIS_{i},d{i},descendant of DIS_0 d0")
    rows.append(f"DIS_{n_dis // 2},d{n_dis // 2},root")
    for i in range(n_dis // 2 + 1, n_dis):
        rows.append(f"DIS_{i},d{i},descendant of DIS_{n_dis // 2} d{n_dis // 2}")
    (seeds / "seed_diseases_neurodeg.csv").write_text("\n".join(rows) + "\n")
    (seeds / "seed_nodes_core.csv").write_text("id,label,name\nGEN_0,GEN,g0\n")
    (seeds / "seed_terms_core.csv").write_text("id,label,name\n")

    import argparse
    args = argparse.Namespace(
        data_dir=str(data), seed_dir=str(seeds), out_dir=str(out),
        max_genes=8, term_genes=5, max_pheno=5, gene_neighbors=5,
        val_frac=0.15, test_frac=0.25, gen_frac=0.0, seed=0, no_prune=False, force=True,
    )
    payload = build(args)
    kept = set(payload["node_ids"])
    if "DIS_EXTRA" in kept or "ANA_0" in kept:
        raise SystemExit(f"pruning kept a distractor: {kept & {'DIS_EXTRA', 'ANA_0'}}")
    if "DRUG_EXTRA" not in kept:
        raise SystemExit("pruning dropped a drug that targets a seed-disease gene")
    missing = [f"DIS_{i}" for i in range(n_dis) if f"DIS_{i}" not in kept]
    if missing:
        raise SystemExit(f"pruning dropped seed diseases: {missing[:5]}")
    if "DRG-DIS|INDICATION" not in payload["rel_names"]:
        raise SystemExit(f"indication relation missing: {payload['rel_names']}")
    if payload["split"]["val"].numel() == 0 or payload["split"]["test"].numel() == 0:
        raise SystemExit("prep produced an empty val or test split")
    print("prep ok", {k: int(v.numel()) for k, v in payload["split"].items()},
          "nodes", payload["num_nodes"], "shared", round(payload["shared_gene_frac"], 3))

    args = parse_args([
        "--split", str(out / "okg_split.pt"),
        "--out_dir", str(out),
        "--tag", "prep",
        "--steps", "200",
        "--eval_every", "50",
        "--patience", "20",
        "--bs", "32",
        "--dim", "64",
        "--dropout", "0",
        "--edge_drop", "0",
        "--lr", "3e-3",
        "--aux", "0.05",
        "--device", "cpu",
        "--eval_test",
        "--seed", "0",
    ])
    result = train(args)
    val = result["val"]["INDICATION"]
    print("prep-train val", val)
    if val["AUPRC"] < 0.4 or val["MRR"] < 0.35 or val["H10"] < 0.6:
        raise SystemExit(f"pruned synthetic graph did not learn: {val}")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_average_precision()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        synthetic_split(root / "split.pt")
        check_trainer(root / "split.pt", root / "run")
        check_prep(root / "prep")
    print("smoke test passed")
