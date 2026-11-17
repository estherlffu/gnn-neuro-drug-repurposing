import polars as pl, json, collections
pl.Config.set_fmt_str_lengths(60); pl.Config.set_tbl_rows(40); pl.Config.set_tbl_width_chars(150)
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from okg_files import find_table
DATA = Path(os.environ.get("OKG_DATA", "."))
NODES, EDGES = find_table(DATA, "nodes"), find_table(DATA, "edges")
K = os.environ.get("OKG_KEEP", "keep/")
ROOTS = ["MONDO_0004975",  # Alzheimer disease
         "MONDO_0005180",  # Parkinson disease
         "MONDO_0004976",  # amyotrophic lateral sclerosis
         "MONDO_0007739",  # Huntington disease
         "MONDO_0005301",  # multiple sclerosis
         "MONDO_0017276",  # frontotemporal dementia
         "EFO_0006792",    # Lewy body dementia
         "EFO_1001050",    # multiple system atrophy
         "MONDO_0019037",  # progressive supranuclear palsy
         "MONDO_0018926"]  # human prion disease
n = pl.read_parquet(NODES).filter(pl.col("label") == "DIS")
name = {i: json.loads(p).get("name") for i, p in zip(n["id"], n["properties"])}
miss = [r for r in ROOTS if r not in name]
assert not miss, f"roots not in graph: {miss}"

E = pl.scan_parquet(EDGES)
par = E.filter((pl.col("label") == "DIS-DIS") & (pl.col("relation") == "PARENT")).select("from", "to").collect()
ch = collections.defaultdict(list)
for a, b in zip(par["from"], par["to"]):
    ch[a].append(b)                       # from = parent, to = child

src, fams = {}, {}
for r in ROOTS:
    fam, st = {r}, [r]
    while st:
        u = st.pop()
        for v in ch.get(u, ()):
            if v not in fam:
                fam.add(v); st.append(v)
    fams[r] = fam
    for i in fam:
        src.setdefault(i, r)

ids = list(src)
ind = E.filter((pl.col("label") == "DRG-DIS") & (pl.col("relation") == "INDICATION")
               & (pl.col("from").is_in(ids) | pl.col("to").is_in(ids))).select("from", "to").collect()
cnt = collections.Counter(b if b in name else a for a, b in zip(ind["from"], ind["to"]))

print("root family sizes (descendants via PARENT edges):")
for r in ROOTS:
    print(f"  {r} {name[r]}: {len(fams[r])} diseases, INDICATION edges {sum(cnt[i] for i in fams[r])}")

df = pl.DataFrame([(i, name[i], "root" if i in ROOTS else f"descendant of {src[i]} {name[src[i]]}") for i in ids],
                  schema=["id", "name", "source"], orient="row").sort("source", "id")
df.write_csv(K + "seed_diseases_neurodeg.csv")
tot = sum(cnt[i] for i in ids)
print(f"\nseed_diseases_neurodeg.csv: {df.height} diseases | INDICATION edges {tot} (all splits, sizing only) | expected test ~{0.05*tot:.0f}")
nl = set(pl.read_csv(K + "seed_diseases.csv")["id"])
print("overlap with the 16 neuroinflammatory diseases:", len(nl & set(ids)))
