import json
import sys

import polars as pl

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    import optimuskg

    path = optimuskg.get_file("nodes.parquet")

node = (
    pl.scan_parquet(path)
    .filter(pl.col("label") == "DIS")
    .head(1)
    .collect()
    .to_dicts()[0]
)
print("id:", node["id"])
print("label:", node["label"])

properties = node["properties"]
if isinstance(properties, str):
    properties = json.loads(properties)
for key, value in properties.items():
    print(f"{key}: {value}")
