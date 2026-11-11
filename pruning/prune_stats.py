import torch, polars as pl, argparse

p = argparse.ArgumentParser()
p.add_argument("--tier", default="core", choices=["core", "ext"])
p.add_argument("--hops", type=int, default=1)
p.add_argument("--max_deg", type=int, default=2000)
p.add_argument("--exclude", nargs="*", default=["ANA-GEN"])
args = p.parse_args()

D = "/local/scratch/kkim636/.cache/optimuskg/doi_10_7910_DVN_IYNGEV/3.2/"
n = pl.read_parquet(D + "largest_connected_component_nodes.parquet")
n = n.with_columns(pl.col("properties").str.json_path_match("$.name").alias("name"))
n = n.join(pl.read_parquet("node_index.parquet").select("id", "nid"), on="id").sort("nid")
d = torch.load("okg_split.pt")
src, dst, rid, N = d["src"], d["dst"], d["rid"], d["num_nodes"]
rels = pl.read_parquet("rel_index.parquet")
names = dict(zip(rels["rid"].to_list(), rels["rtype"].to_list()))

tr = d["split"]["train"]
s, t, r = src[tr], dst[tr], rid[tr]
ex_ids = torch.tensor([k for k, v in names.items() if any(v.startswith(e) for e in args.exclude)], dtype=torch.long)
ok = ~torch.isin(r, ex_ids)
print("train edges:", s.numel(), "| after excluding", args.exclude, ":", int(ok.sum()))

def mask_of(df):
    m = torch.zeros(N, dtype=torch.bool)
    m[torch.tensor(df["nid"].to_list(), dtype=torch.long)] = True
    return m

core = "(?i)neuroinflam|microglial cell activation|astrocyte activation|NLRP3 inflammasome"
ext = core + "|toll-like receptor signaling|interleukin-1 signaling|interleukin-6 signaling|microglia"
terms = n.filter(pl.col("label").is_in(["BPO", "PWY"]) & pl.col("name").str.contains(core if args.tier == "core" else ext))
is_term = mask_of(terms)
is_gen = mask_of(n.filter(pl.col("label") == "GEN"))
is_neuro = mask_of(n.filter((pl.col("label") == "DIS") & pl.col("properties").str.contains("EFO_0000618")))

seed = torch.zeros(N, dtype=torch.bool)
seed[t[is_term[s] & is_gen[t]]] = True
seed[s[is_gen[s] & is_term[t]]] = True
print(f"tier={args.tier}: {int(is_term.sum())} term nodes, {int(seed.sum())} unique seed genes")

import os
os.makedirs("keep", exist_ok=True)
terms.select("id", "label", "name").sort("id").write_csv(f"keep/seed_terms_{args.tier}.csv")
n.filter(pl.Series(seed.numpy())).select("id", "label", "name").sort("id").write_csv(f"keep/seed_nodes_{args.tier}.csv")
s2, t2, r2 = s[ok], t[ok], r[ok]
deg = torch.bincount(s2, minlength=N) + torch.bincount(t2, minlength=N)
keep = is_term | seed
new = keep.clone()

ind_rid = [k for k, v in names.items() if v == "DRG-DIS|INDICATION"][0]
ind_all = int((r == ind_rid).sum())

def report(tag):
    inside = keep[s2] & keep[t2]
    n_in = int(inside.sum())
    ind = int(((r2 == ind_rid) & inside).sum())
    print(f"\n[{tag}] nodes {int(keep.sum())} ({100*keep.float().mean():.1f}%), edges {n_in} ({100*n_in/s.numel():.1f}% of train)")
    print("  neuro-disease nodes kept:", int((keep & is_neuro).sum()), "| DRG-DIS|INDICATION train edges kept:", ind, "/", ind_all)
    print(n.filter(pl.Series(keep.numpy())).group_by("label").len().sort("len", descending=True))

report("seed+terms")
for h in range(1, args.hops + 1):
    can = new & (deg <= args.max_deg)
    add = torch.zeros(N, dtype=torch.bool)
    add[t2[can[s2]]] = True
    add[s2[can[t2]]] = True
    new = add & ~keep
    keep |= add
    report(f"hop {h}")

out = f"keep_nodes_{args.tier}_h{args.hops}.parquet"
n.filter(pl.Series(keep.numpy())).select("id").write_parquet(out)
print("\nsaved", out)
