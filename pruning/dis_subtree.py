import polars as pl, json, sys, collections, os
from collections import deque
pl.Config.set_fmt_str_lengths(55); pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(170)
D = "/local/scratch/kkim636/.cache/optimuskg/doi_10_7910_DVN_IYNGEV/3.2/"
K = "/local/scratch/kkim636/keep/"
ROOTS = sys.argv[1:] or ["EFO_0020092", "MONDO_0002977", "MONDO_0019956"]
n = pl.read_parquet(D + "largest_connected_component_nodes.parquet").filter(pl.col("label") == "DIS")
P = {i: json.loads(p) for i, p in zip(n["id"], n["properties"])}
name = {i: p.get("name") for i, p in P.items()}
E = pl.scan_parquet(D + "largest_connected_component_edges.parquet")
par = E.filter((pl.col("label") == "DIS-DIS") & (pl.col("relation") == "PARENT")).select("from", "to").collect()
ch = collections.defaultdict(list)
for a, b in zip(par["from"], par["to"]):
    ch[a].append(b)                      # from = parent, to = child

def subtree(root):
    depth = {root: 0}; q = deque([root])
    while q:
        u = q.popleft()
        for v in ch.get(u, ()):
            if v not in depth:
                depth[v] = depth[u] + 1; q.append(v)
    return depth

subs = {r: subtree(r) for r in ROOTS}
allids = sorted(set().union(*[set(s) for s in subs.values()]))
ind = E.filter((pl.col("label") == "DRG-DIS") & (pl.col("relation") == "INDICATION")
               & (pl.col("from").is_in(allids) | pl.col("to").is_in(allids))).select("from", "to").collect()
dg = E.filter((pl.col("label") == "DIS-GEN") & (pl.col("from").is_in(allids) | pl.col("to").is_in(allids))).select("from", "to").collect()
indc = collections.Counter(b if b in P else a for a, b in zip(ind["from"], ind["to"]))
genc = collections.Counter(a if a in P else b for a, b in zip(dg["from"], dg["to"]))

cand = pl.read_csv(K + "neuroinflam_dis_candidates.csv").select("id", "n_flags") if os.path.exists(K + "neuroinflam_dis_candidates.csv") else None
flags = dict(zip(cand["id"], cand["n_flags"])) if cand is not None else {}
review = {i for i, f in flags.items() if f >= 2}

main = subs[ROOTS[0]]
rows = [(depth, i, name.get(i), indc[i], genc[i], "EFO_0000618" in (P[i].get("therapeutic_areas") or []), flags.get(i))
        for i, depth in main.items()]
tab = pl.DataFrame(rows, schema=["depth", "id", "name", "indications", "gene_edges", "neuro_tag", "n_flags"], orient="row").sort(["depth", "indications"], descending=[False, True])
print(f"== subtree of {ROOTS[0]} {name[ROOTS[0]]}: {tab.height} diseases (incl. root), max depth {tab['depth'].max()}")
print("   total INDICATION edges:", int(tab["indications"].sum()), "| diseases with >=1 indication:", tab.filter(pl.col("indications") > 0).height)
print(tab)

print("\nsubtree sizes and overlap with the first root's subtree:")
for r in ROOTS:
    s = subs[r]
    print(f"  {r} {name.get(r)}: {len(s)} diseases, INDICATION {sum(indc[i] for i in s)}, overlap with main {len(set(s) & set(main))}")

print(f"\nreview set (n_flags>=2): {len(review)} | in subtree: {len(review & set(main))}")
print("subtree members NOT in the keyword candidate file:", [(i, name[i]) for i in main if i not in flags][:30])
print("review-set diseases NOT in the subtree (first 25):", [(i, name[i]) for i in review if i not in main][:25])

out = tab.with_columns(pl.lit(None, dtype=pl.Int8).alias("include"), pl.lit("").alias("note"))
out.write_csv(K + f"dis_subtree_{ROOTS[0]}.csv")
print("saved keep/dis_subtree_%s.csv" % ROOTS[0])
