import torch, polars as pl, argparse

p = argparse.ArgumentParser()
p.add_argument("--keep", nargs="*", default=[])
p.add_argument("--cap", type=int, default=2000)
p.add_argument("--k", type=int, default=50)
p.add_argument("--out", default="/local/scratch/kkim636/evalset.pt")
args = p.parse_args()

O = "/local/scratch/kkim636/"
D = O + ".cache/optimuskg/doi_10_7910_DVN_IYNGEV/3.2/"
d = torch.load(O + "okg_split.pt")
src, dst, rid = d["src"], d["dst"], d["rid"]
N, R, T = d["num_nodes"], d["num_rels"], d["num_ntypes"]
node_type = d["node_type"]
idx = pl.read_parquet(O + "node_index.parquet").select("id", "nid")

def to_mask(nids):
    m = torch.zeros(N, dtype=torch.bool)
    m[torch.from_numpy(nids.astype("int64"))] = True
    return m

U = torch.ones(N, dtype=torch.bool)
for f in args.keep:
    ids = pl.read_parquet(f).select("id").unique()
    U &= to_mask(idx.join(ids, on="id")["nid"].to_numpy())
print("common universe U:", int(U.sum()), "nodes")

nodes = pl.read_parquet(D + "largest_connected_component_nodes.parquet").join(idx, on="id")
is_neuro = to_mask(nodes.filter((pl.col("label") == "DIS") &
                                pl.col("properties").str.contains("EFO_0000618"))["nid"].to_numpy())

rels = pl.read_parquet(O + "rel_index.parquet")
ind_rid = rels.filter(pl.col("rtype") == "DRG-DIS|INDICATION")["rid"][0]

all_keys = (src * N + dst) * R + rid     # known true edges (for filtered negatives)

un = U.nonzero().squeeze(1); ut = node_type[un]
u_sorted = un[torch.argsort(ut)]
cnt = torch.bincount(ut, minlength=T)
start = torch.cumsum(cnt, 0) - cnt

g = torch.Generator().manual_seed(123)
out = {"U": U, "k": args.k}
for name in ["val", "test"]:
    ie = d["split"][name]
    s, t, r = src[ie], dst[ie], rid[ie]
    m = U[s] & U[t]
    s, t, r = s[m], t[m], r[m]
    sel = []
    for rr in torch.unique(r).tolist():
        ii = (r == rr).nonzero().squeeze(1)
        sel.append(ii[torch.randperm(ii.numel(), generator=g)[:args.cap]])
    sel = torch.cat(sel)
    s, t, r = s[sel], t[sel], r[sel]

    tt = node_type[t]
    rr_ = (torch.rand(t.numel(), args.k, generator=g) * cnt[tt].unsqueeze(1)).long()
    rr_ = torch.minimum(rr_, cnt[tt].unsqueeze(1) - 1)
    neg = u_sorted[start[tt].unsqueeze(1) + rr_]

    k1 = ((s.unsqueeze(1) * N + neg) * R + r.unsqueeze(1)).flatten()
    k2 = ((neg * N + s.unsqueeze(1)) * R + r.unsqueeze(1)).flatten()
    bad = (torch.isin(k1, all_keys) | torch.isin(k2, all_keys)).view_as(neg)
    valid = ~bad & (neg != t.unsqueeze(1))

    out[name] = {"s": s, "t": t, "r": r, "neg": neg, "valid": valid, "tail_neuro": is_neuro[t]}
    ind = r == ind_rid
    print(f"[{name}] edges {s.numel()} | INDICATION {int(ind.sum())} | INDICATION w/ neuro tail {int((ind & is_neuro[t]).sum())}"
          f" | valid negatives {valid.float().mean():.4f}")
    if int(ind.sum()) < 500:
        print("  WARNING: few INDICATION edges in U -> drug-disease evaluation will be noisy")

torch.save(out, args.out)
print("saved", args.out)
