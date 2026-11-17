"""Find OptimusKG parquet tables.

The training graph for this repo is the full export, `nodes.parquet` and
`edges.parquet`. The largest-connected-component files are only a fallback.
"""
from pathlib import Path

NODE_NAMES = ("nodes.parquet", "largest_connected_component_nodes.parquet")
EDGE_NAMES = ("edges.parquet", "largest_connected_component_edges.parquet")


def find_table(data_dir, kind):
    data_dir = Path(data_dir)
    names = NODE_NAMES if kind == "nodes" else EDGE_NAMES
    for name in names:
        path = data_dir / name
        if path.exists():
            if name != names[0]:
                print(f"using {path} ({names[0]} was not in {data_dir})", flush=True)
            return path
    tried = ", ".join(str(data_dir / name) for name in names)
    raise SystemExit(f"could not find the {kind} table. Looked for: {tried}")
