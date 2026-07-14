"""
fde_deduplication.py

Stage 3: Semantic deduplication of extracted FDE (Fisheries Dispute Event) records.

Reads all fde_events_*.csv files produced by Stage 2 (fde_data_collection.py)
from <output>/stage2/, concatenates them, decodes coded columns, and removes
duplicate FDE records using a combination of hard filters (country, date window,
violence score) and semantic similarity on location + actors + issue_note.

Output is written to <output>/stage3/final_fde_output.csv.

Deduplication model
-------------------
  Sentence-transformers model: sentence-transformers/all-MiniLM-L6-v2
  Maps text to a 384-dim vector space; cosine similarity used for comparison.

Duplicate criteria (all must hold)
------------------------------------
  1. Same country
  2. Dates within `date_window` days of each other (default: 7)
  3. Same ViolenceScore
  4. Semantic similarity of "<location>, <actor1>, <actor2>, <issue_note>"
     strings >= `sim_threshold` (default: 0.65)

When two records are flagged as duplicates, the one with the lower article_id
(earlier in the catalog) is kept as the canonical record.

Pipeline integration (run_pipeline.py)
---------------------------------------
    from fde_deduplication import run_deduplication

    run_deduplication(args, stage2_dir=output / "stage2", stage3_dir=output / "stage3")

Standalone usage — import
-------------------------
    from fde_deduplication import load_and_concat, flag_duplicates, deduplicate

    df     = load_and_concat("results/stage2/")
    deduped = deduplicate(df, date_window=7, sim_threshold=0.65)
    deduped.to_csv("results/stage3/final_fde_output.csv", index=False)

Standalone usage — CLI
----------------------
    python fde_deduplication.py --output results/
    python fde_deduplication.py --output results/ \\
        --date-window 14 --sim-threshold 0.70
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column-decode dictionaries  (match Stage 2 schema)
# ---------------------------------------------------------------------------

LOC_ACC_DICT: dict[int, str] = {
    0: "Country",
    1: "District",
    2: "International",
    3: "NA",
    4: "Region",
    5: "Town",
}

REGION_DICT: dict[int, str] = {
    0: "Arctic",
    1: "Horn of Africa",
    2: "Oceania",
    3: "South America",
    4: "South Asia",
}

# FDE criteria columns to remove
FDE_CRITERIA_COLS: list[str] = [
    "fde_min_two_actors",
    "fde_discrete_temporal_moment",
    "fde_discrete_location",
    "fde_fisheries_resource_contested",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_concat(stage2_dir: Path | str) -> pd.DataFrame:
    """
    Load all fde_events_*.csv files from `stage2_dir`, concatenate them,
    and return a single cleaned DataFrame.

    Cross-file deduplication is by article_id — if the same article_id
    appears in multiple files (e.g. from overlapping pipeline runs), only
    the first occurrence is kept.

    Parameters
    ----------
    stage2_dir : Path | str
        Directory containing one or more fde_events_*.csv files
        (i.e. <output>/stage2/).

    Returns
    -------
    pd.DataFrame
        Concatenated, sorted, and decoded FDE records.
    """
    stage2_dir = Path(stage2_dir)
    csv_files = sorted(stage2_dir.glob("fde_events*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No fde_events_*.csv files found in {stage2_dir}"
        )

    log.info(f"  Found {len(csv_files)} Stage 2 CSV file(s)")

    df_base: pd.DataFrame | None = None
    for fname in csv_files:
        df_new = pd.read_csv(fname)
        if df_base is None:
            df_base = df_new
        else:
            # Keep rows whose article_id is not already present
            add_rows = df_new[~df_new["article_id"].isin(df_base["article_id"])]
            df_base = pd.concat([df_base, add_rows], ignore_index=True)

    # Normalize article_id to integer for consistent sorting
    df_base["article_id"] = (
        df_base["article_id"].astype(str).str.split("_").str[-1].astype(int)
    )
    df_base = df_base.sort_values("article_id").reset_index(drop=True)

    # Decode date
    df_base["date"] = pd.to_datetime(df_base["date"], errors="coerce")

    # Decode coded columns
    df_base["loc_accuracy"] = df_base["loc_accuracy"].map(LOC_ACC_DICT)
    df_base["region"]       = df_base["region"].map(REGION_DICT)

    # Drop FDE criteria flags if present (intermediate flags, not outputs)
    cols_to_drop = [c for c in FDE_CRITERIA_COLS if c in df_base.columns]
    if cols_to_drop:
        df_base.drop(columns=cols_to_drop, inplace=True)

    log.info(f"  Total records after concatenation: {len(df_base)}")
    return df_base


# ---------------------------------------------------------------------------
# Semantic similarity
# ---------------------------------------------------------------------------

def _load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """Load the sentence-transformer embedding model."""
    log.info(f"  Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def semantic_similarity(s1: str, s2: str, model: SentenceTransformer) -> float:
    """
    Return cosine similarity (0–1) between two strings.

    Parameters
    ----------
    s1, s2 : str
    model : SentenceTransformer

    Returns
    -------
    float
    """
    emb = model.encode([s1, s2])
    return cosine_similarity([emb[0]], [emb[1]]).item()


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def _build_text_string(row: pd.Series) -> str:
    """
    Build the composite text string used for semantic comparison.

    Handles both column names (location/actor1/actor2) and any 
    variants (Location/Actor1/Actor2) for robustness.
    """
    original = ["location", "actor1", "actor2", "issue_note"]
    renamed  = ["Location", "Actor1", "Actor2", "issue_note"]

    if all(c in row.index for c in original):
        return f"{row['location']}, {row['actor1']}, {row['actor2']}, {row['issue_note']}"
    elif all(c in row.index for c in renamed):
        return f"{row['Location']}, {row['Actor1']}, {row['Actor2']}, {row['issue_note']}"
    else:
        # Fallback: tolerate either naming convention per field
        loc    = row.get("location", row.get("Location", ""))
        actor1 = row.get("actor1",   row.get("Actor1", ""))
        actor2 = row.get("actor2",   row.get("Actor2", ""))
        note   = row.get("issue_note", "")
        return f"{loc}, {actor1}, {actor2}, {note}"


def are_duplicates(
    r1: pd.Series,
    r2: pd.Series,
    model: SentenceTransformer,
    date_window: int = 7, # Maximum days between event dates to be considered a duplicate.
    sim_threshold: float = 0.65, # Minimum cosine similarity (0–1)
) -> bool:
    """
    Return True if two FDE records are likely describing the same event.
    Hard filters are applied first; semantic similarity is checked
    last only when all hard filters pass.
    """
    # 1. Country must match exactly
    if r1["country"] != r2["country"]:
        return False

    # 2. Date proximity
    try:
        d1 = pd.to_datetime(r1["date"])
        d2 = pd.to_datetime(r2["date"])
        if abs((d1 - d2).days) > date_window:
            return False
    except Exception:
        pass  # if dates are unparseable, skip this filter

    # 3. ViolenceScore must match (handle both column name variants)
    vs_col = "ViolenceScore" if "ViolenceScore" in r1.index else "violence_score"
    if vs_col in r1.index and r1[vs_col] != r2[vs_col]:
        return False

    # 4. Semantic similarity of composite text (expensive)
    s1 = _build_text_string(r1)
    s2 = _build_text_string(r2)
    if semantic_similarity(s1, s2, model) < sim_threshold:
        return False

    return True


def flag_duplicates(
    df: pd.DataFrame, # Cleaned FDE DataFrame from load_and_concat().
    model: SentenceTransformer,
    date_window: int = 7,
    sim_threshold: float = 0.65,
) -> pd.DataFrame:
    """
    Add a `duplicate_of` column identifying the duplicated article_id

    The record with the lower article_id is kept as canonical; 
    later duplicates are marked.
    """
    df = df.copy()
    df["duplicate_of"] = None
    ids = df["article_id"].tolist()
    n   = len(df)

    total_pairs = n * (n - 1) // 2
    log.info(f"  Checking {total_pairs:,} pairs across {n} records …")

    for i, j in tqdm(combinations(range(n), 2), total=total_pairs, desc="Dedup"):
        # Skip j if it's already been marked as a duplicate
        if df.iloc[j]["duplicate_of"] is not None:
            continue
        if are_duplicates(df.iloc[i], df.iloc[j], model, date_window, sim_threshold):
            df.at[df.index[j], "duplicate_of"] = ids[i]

    n_dups = df["duplicate_of"].notna().sum()
    log.info(f"  Flagged {n_dups} duplicate(s) → {n - n_dups} canonical record(s) remain")
    return df


def deduplicate(
    df: pd.DataFrame,
    model: SentenceTransformer | None = None,
    date_window: int = 7,
    sim_threshold: float = 0.65,
) -> pd.DataFrame:
    """
    Run the full deduplication pipeline on an already-loaded DataFrame.
    Duplicate rows removed, index reset.
    """
    if model is None:
        model = _load_embedding_model()

    flagged = flag_duplicates(df, model, date_window, sim_threshold)
    return df[flagged["duplicate_of"].isna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Pipeline stage runner  (called by run_pipeline.py)
# ---------------------------------------------------------------------------

def run_deduplication(
    args,
    stage2_dir: Path,
    stage3_dir: Path,
) -> Path:
    """
    Reads all fde_events_*.csv files from `stage2_dir`, deduplicates them,
    and writes final_fde_output.csv to `stage3_dir`.

    Parameters
    ----------
    args : argparse.Namespace
        Pipeline args. Reads: args.date_window, args.sim_threshold,
    stage2_dir : Path
        Output directory from Stage 2 (<output>/stage2/).
    stage3_dir : Path
        Output directory for Stage 3 (<output>/stage3/).

    Returns
    -------
    Path
        Path to the written final_fde_output.csv.
    """

    stage3_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log.info(f"  Loading Stage 2 output: {stage2_dir}")
    
    df    = load_and_concat(stage2_dir)
    model = _load_embedding_model(
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )

    deduped = deduplicate(
        df,
        model         = model,
        date_window   = getattr(args, "date_window",   7),
        sim_threshold = getattr(args, "sim_threshold", 0.65),
    )

    out_path = stage3_dir / "final_fde_output.csv"
    deduped.to_csv(out_path, index=False)

    log.info(f"Stage 3 complete  ({time.time() - t0:.1f}s)")
    log.info(f"  Records before dedup : {len(df)}")
    log.info(f"  Records after dedup  : {len(deduped)}")
    log.info(f"  Output               : {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI  (standalone use, mirrors run_pipeline.py conventions)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: Semantic deduplication of FDE CSV outputs.\n\n"
            "Reads fde_events_*.csv from <output>/stage2/, removes duplicate\n"
            "events, and writes final_fde_output.csv to <output>/stage3/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results"),
        help="Pipeline root output directory (default: results/)"
    )
    parser.add_argument(
        "--date-window", type=int, default=7, dest="date_window",
        help="Max days between event dates to be considered duplicates (default: 7)"
    )
    parser.add_argument(
        "--sim-threshold", type=float, default=0.65, dest="sim_threshold",
        help="Minimum cosine similarity to flag as duplicate (default: 0.65)"
    )

    return parser


def main() -> None:

    args = _build_parser().parse_args()

    log.info("=" * 60)
    log.info("Semantic Deduplication")
    log.info("=" * 60)

    stage2_dir = args.output / "stage2"
    stage3_dir = args.output / "stage3"

    run_deduplication(args, stage2_dir=stage2_dir, stage3_dir=stage3_dir)


if __name__ == "__main__":
    main()
