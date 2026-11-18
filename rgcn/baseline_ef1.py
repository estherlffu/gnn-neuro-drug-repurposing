
"""Build a fully unpruned OptimusKG graph with an indication-only 90/5/5 split.

Baseline relative to pruning_ef3.py:
- Retains ALL edges from the allow-listed relation classes.
- Does not apply seed-anchor filtering.
- Does not apply seed-neighborhood node selection.
- Does not apply the gene-gene neighbor cap.

Uses the same 79-disease seed set, relation definitions, indication-only
prediction targets, and pair-level random split logic as pruning_ef3.py.

Contraindication edges remain in the graph artifact as context/bookkeeping,
but are not prediction targets.
"""

import sys
from pathlib import Path

import polars as pl

# Make the rgcn directory available for importing pruning_ef3.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pruning_ef3 as prep


def load_edges_unpruned(edge_path, anchor_ids):
    """Load every edge belonging to an allow-listed relation class.

    Unlike pruning_ef3.load_edges(), this function does NOT restrict large
    relation classes to edges touching seed/anchor IDs.
    """

    scan = pl.scan_parquet(edge_path)

    try:
        columns = scan.collect_schema().names()
    except AttributeError:
        columns = scan.schema.names()

    wanted = [
        c for c in ("from", "to", "label", "relation", "undirected")
        if c in columns
    ]

    missing = {"from", "to", "label", "relation"} - set(wanted)

    if missing:
        raise SystemExit(
            f"{edge_path} is missing columns: {sorted(missing)}"
        )

    frame = (
        pl.scan_parquet(edge_path)
        .select(wanted)
        .with_columns(
            pl.col("from").cast(pl.Utf8),
            pl.col("to").cast(pl.Utf8),
            pl.col("label").cast(pl.Utf8),
            pl.col("relation").cast(pl.Utf8),
        )
        .filter(pl.col("label").is_in(sorted(prep.ALLOW)))
        .collect()
    )

    if frame.height == 0:
        labels = (
            pl.scan_parquet(edge_path)
            .select(pl.col("label").unique())
            .collect()["label"]
            .to_list()
        )
        raise SystemExit(
            f"No allow-listed edges in {edge_path}. "
            f"Labels present: {labels[:40]}"
        )

    if "undirected" not in frame.columns:
        frame = frame.with_columns(
            pl.lit(False).alias("undirected")
        )

    frame = frame.with_columns(
        pl.col("undirected").fill_null(False)
    )

    print(
        f"UNPRUNED BASELINE: retaining all "
        f"{frame.height:,} allow-listed edges.",
        flush=True,
    )

    return frame


def main(argv=None):
    args = prep.parse_args(argv)

    # Require the unpruned graph-building branch.
    args.no_prune = True

    # Replace the original anchor-filtered edge loader.
    prep.load_edges = load_edges_unpruned

    print("=" * 65, flush=True)
    print("BUILDING UNPRUNED OPTIMUSKG BASELINE", flush=True)
    print("=" * 65, flush=True)

    print(f"Data directory: {args.data_dir}", flush=True)
    print(f"Seed directory: {args.seed_dir}", flush=True)
    print(f"Output directory: {args.out_dir}", flush=True)
    print("Anchor filtering: DISABLED", flush=True)
    print("Neighborhood pruning: DISABLED", flush=True)
    print("Gene-neighbor cap: DISABLED", flush=True)
    print("Prediction task: DRG-DIS|INDICATION", flush=True)
    print("Split: random pair-level 90/5/5", flush=True)

    # Use the original EF3 build function so the graph artifact has
    # exactly the same schema expected by the R-GCN training script.
    result = prep.build(args)

    print("=" * 65, flush=True)
    print("UNPRUNED BASELINE GRAPH BUILD COMPLETE", flush=True)
    print(f"Nodes: {result['num_nodes']:,}", flush=True)
    print(f"Relations: {result['num_rels']:,}", flush=True)
    print(f"Output: {Path(args.out_dir) / 'okg_split.pt'}", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
