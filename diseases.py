import sys

import polars as pl

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    import optimuskg

    path = optimuskg.get_file("nodes.parquet")

diseases = (
    pl.scan_parquet(path)
    .filter(pl.col("label") == "DIS")
    .select("id", pl.col("properties").str.json_path_match("$.name").alias("name"))
    .unique(subset=["id"])
    .sort("name")
    .collect()
)
diseases.write_csv("diseases.csv")
print(diseases)
