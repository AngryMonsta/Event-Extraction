"""
run_pipeline.py
---------------
End-to-end orchestrator for the Fisheries Conflict pipeline.

Stages
------
  Stage 0  — Text extraction   (text_extraction.py)
  Stage 1  — Dispute detection (dispute_detection.py)
  Stage 2  — FDE extraction    (fde_data_collection.py)

Usage
-----
    python run_pipeline.py [options]

    # Minimal — ZIP input, auto device
    python run_pipeline.py \\
        --zip       data/Somalia.ZIP \\
        --output    results/
        
    # Single raw article string supplied inline on the CLI
    python run_pipeline.py \\
        --string    "Kenya coastguard arrested 12 Somali fishers near Lamu on 3 March 2005..." \\
        --output    results/
 
    # Batch of raw strings from a JSON file  (array of strings, or array of
    # {"doc_id": "...", "text": "..."} objects for explicit IDs)
    python run_pipeline.py \\
        --strings-json  data/articles.json \\
        --output        results/

    # Full control
    python run_pipeline.py \\
        --zip           data/Somalia.ZIP \\
        --output        results/ \\
        --model         google/gemma-4-E4B-it \\
        --device        cuda:0 \\
        --device-map-mode single \\
        --max-new-tokens  2048 \\
        --temperature     0.1 \\
        --checkpoint-every 50 \\
        --batch-size      4 \\
        --id-start        1 \\
        --limit-s1        None \\
        --limit-s2        None \\
        --skip-stage0         \\   # re-use already-extracted JSON files
        --skip-stage1             # re-use existing Stage 1 output JSON

    # Skip extraction (articles already in results/stage0/)
    python run_pipeline.py \\
        --folder    data/articles/ \\
        --output    results/ \\
        --skip-stage0

    # Resume Stage 2 from a previous Stage 1 output
    python run_pipeline.py \\
        --output       results/ \\
        --stage1-file  results/stage1/detection_output.json \\
        --skip-stage0 --skip-stage1

Directory layout created under --output
----------------------------------------
    <output>/
      stage0/   <- Stage 0 JSON files  (eDOC_XXXX.json)
      stage1/   <- Stage 1 detection JSON + checkpoint
      stage2/   <- Stage 2 FDE JSONL, CSV, failed log, checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Stage modules (relative imports — works from repo root after pip install -e .) ──
from . import text_extraction      as _stage0_mod
from . import dispute_detection    as _stage1_mod
from . import fde_data_collection  as _stage2_mod

# ── Shared model registry ─────────────────────────────────────────────────
from .model_registry import load_shared_model


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Silence noisy third-party loggers (httpx, HuggingFace Hub HTTP requests, etc.)
for _noisy in (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "huggingface_hub.utils",
    "transformers",
    "accelerate",
    "filelock",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Stage 0 runner
# ---------------------------------------------------------------------------

def run_stage0(args, text_dir: Path) -> int:
    """
    Extract articles to JSON files in text_dir.
    Returns the number of articles written.
 
    Supported input modes
    ---------------------
    --zip            ZIP archive of LexisNexis .docx files
    --folder         Directory of .docx / .doc / .txt files
    --file           Single .docx / .doc / .txt file
    --string         One raw article string passed inline on the CLI
    --strings-json   Path to a JSON file containing either:
                       • a plain array of strings  ["text1", "text2", ...]
                       • an array of objects        [{"doc_id": "X", "text": "..."}, ...]
                     Objects may omit "doc_id"; the position index is used as a
                     fallback so IDs are always unique.
    """
    log.info("=" * 60)
    log.info("STAGE 0 — Text Extraction")
    log.info("=" * 60)
 
    text_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
 
    if args.zip:
        count = _stage0_mod.process_zip(
            zip_path   = str(args.zip),
            output_dir = str(text_dir),
            id_start   = args.id_start,
        )
 
    elif args.folder:
        count = _stage0_mod.process_folder(
            folder_path = str(args.folder),
            output_dir  = str(text_dir),
            id_start    = args.id_start,
        )
 
    elif args.file:
        _stage0_mod.process_file(
            path       = str(args.file),
            output_dir = str(text_dir),
        )
        count = 1
 
    elif args.string:
        # Single inline string — use --id-start as the numeric part of the ID.
        doc_id = f"DOC_{args.id_start:04d}"
        _stage0_mod.process_string(
            text       = args.string,
            doc_id     = doc_id,
            output_dir = str(text_dir),
        )
        log.info(f"  Processed inline string → {doc_id}")
        count = 1
 
    elif args.strings_json:
        # Batch of raw strings from a JSON file.
        strings_path = Path(args.strings_json)
        if not strings_path.exists():
            log.error(f"--strings-json file not found: {strings_path}")
            sys.exit(1)
 
        with strings_path.open(encoding="utf-8") as fh:
            raw_input = json.load(fh)
 
        if not isinstance(raw_input, list):
            log.error(
                f"--strings-json must contain a JSON array; "
                f"got {type(raw_input).__name__}"
            )
            sys.exit(1)
 
        count = 0
        for i, item in enumerate(raw_input):
            # Accept both plain strings and {"doc_id": ..., "text": ...} objects.
            if isinstance(item, str):
                text   = item
                doc_id = f"DOC_{(args.id_start + i):04d}"
            elif isinstance(item, dict):
                text = item.get("text", "")
                if not text:
                    log.warning(f"  Item {i}: missing 'text' key — skipping")
                    continue
                # Honour an explicit doc_id when provided, otherwise auto-assign.
                explicit_id = item.get("doc_id")
                doc_id = str(explicit_id) if explicit_id else f"DOC_{(args.id_start + i):04d}"
            else:
                log.warning(
                    f"  Item {i}: expected str or dict, got "
                    f"{type(item).__name__} — skipping"
                )
                continue
 
            _stage0_mod.process_string(
                text       = text,
                doc_id     = doc_id,
                output_dir = str(text_dir),
            )
            log.info(f"  [{i+1}/{len(raw_input)}] processed → {doc_id}")
            count += 1
 
    else:
        log.error(
            "Stage 0: no input provided. "
            "Use --zip / --folder / --file / --string / --strings-json"
        )
        sys.exit(1)
 
    log.info(f"Stage 0 complete — {count} articles written  ({time.time()-t0:.1f}s)")
    return count


# ---------------------------------------------------------------------------
# Stage 1 runner
# ---------------------------------------------------------------------------

def run_stage1(
    args,
    text_dir:    Path,
    stage1_dir:  Path,
    shared_model,
) -> Path:
    """
    Run dispute detection over every eDOC_*.json file in text_dir.
    Returns the path to the completed Stage 1 JSON output file.
    """
    log.info("=" * 60)
    log.info("STAGE 1 — Dispute Detection")
    log.info("=" * 60)

    stage1_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Locate eDOC files
    extraction_files = sorted(
        f for f in text_dir.iterdir()
        if f.name.startswith("eDOC_") and f.suffix == ".json"
    )
    if not extraction_files:
        log.error(f"Stage 1: no eDOC_*.json files found in {text_dir}")
        sys.exit(1)

    if args.limit_s1:
        extraction_files = extraction_files[: args.limit_s1]
        log.info(f"  --limit-s1 active — processing first {args.limit_s1} files")

    log.info(f"  Articles to classify: {len(extraction_files)}")

    # Build system prompt using Stage 1 helper
    from pydantic import TypeAdapter
    schema_json    = json.dumps(
        TypeAdapter(_stage1_mod.DetectionSchema).json_schema(), indent=2
    )
    system_prompt  = _stage1_mod.build_system_prompt(schema_json)

    # Output paths
    run_ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path   = stage1_dir / f"detection_output_{run_ts}.json"
    checkpoint_path = stage1_dir / f"detection_checkpoint_{run_ts}.json"

    all_detections: list[dict] = []
    processed_ids:  set[str]   = set()
    error_count = 0
    pending_batch: list[dict] = []

    # ── Resume from checkpoint if present ─────────────────────────────────
    existing_checkpoints = sorted(stage1_dir.glob("detection_checkpoint_*.json"))
    if existing_checkpoints:
        latest = existing_checkpoints[-1]
        log.info(f"  Resuming Stage 1 from checkpoint: {latest.name}")
        with latest.open() as fh:
            saved = json.load(fh)
        all_detections = saved
        processed_ids  = {e.get("article_id") for e in saved if isinstance(e, dict) and e.get("article_id")}
        error_count    = sum(1 for e in saved if isinstance(e, dict) and "detection_error" in e)
        log.info(f"  {len(all_detections)} entries already processed")

    # ── Per-article detection ──────────────────────────────────────────────
    for file_path in extraction_files:
        article_id = file_path.stem          # "eDOC_0001"
        if article_id in processed_ids:
            log.info(f"  {article_id}: skipped (already in checkpoint)")
            continue

        with file_path.open(encoding="utf-8") as fh:
            extracted_content = json.load(fh)

        result = _stage1_mod.run_detection(
            extracted_content = extracted_content,
            processor         = shared_model.processor,
            model             = shared_model.model,
            system_prompt     = system_prompt,
            max_new_tokens    = args.max_new_tokens,
            temperature       = args.temperature,
        )

        if isinstance(result, _stage1_mod.DetectionSchema):
            entry  = _stage1_mod.OutputFileSchema(
                article_id = article_id,
                detection  = [result],
            ).model_dump(mode="json")
            status = (
                f"fde={result.contains_fde}  fda={result.contains_fda}  "
                f"n={result.num_candidate_events}  conf={result.confidence}"
            )
        else:
            entry  = {"article_id": article_id, "detection_error": result}
            status = f"ERROR — {result.get('error')}"
            error_count += 1

        all_detections.append(entry)
        pending_batch.append(entry)
        processed_ids.add(article_id)
        log.info(f"  {article_id}: {status}")

        if len(pending_batch) >= args.checkpoint_every:
            _stage1_mod._atomic_json_dump(checkpoint_path, all_detections)
            log.info(
                f"  Checkpoint saved → {checkpoint_path.name} "
                f"({len(all_detections)} entries)"
            )
            pending_batch.clear()

    # Final checkpoint + summary write
    if pending_batch:
        _stage1_mod._atomic_json_dump(checkpoint_path, all_detections)

    _stage1_mod._atomic_json_dump(summary_path, all_detections)

    successes = [e for e in all_detections if "detection" in e]
    fde_hits  = sum(1 for e in successes if e["detection"][0]["contains_fde"])
    fda_hits  = sum(1 for e in successes if e["detection"][0]["contains_fda"])

    log.info(f"Stage 1 complete  ({time.time()-t0:.1f}s)")
    log.info(f"  Total classified : {len(all_detections)}")
    log.info(f"  Errors           : {error_count}")
    log.info(f"  contains_fde=T   : {fde_hits}")
    log.info(f"  contains_fda=T   : {fda_hits}")
    log.info(f"  Output           : {summary_path}")

    return summary_path


# ---------------------------------------------------------------------------
# Stage 2 runner
# ---------------------------------------------------------------------------

def run_stage2(
    args,
    stage1_path: Path,
    text_dir:    Path,
    stage2_dir:  Path,
    shared_model,
) -> None:
    """
    Run FDE extraction over FDE-positive articles from Stage 1 output.
    """
    import csv

    log.info("=" * 60)
    log.info("STAGE 2 — FDE Extraction")
    log.info("=" * 60)

    stage2_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Load Stage 1 output ────────────────────────────────────────────────
    log.info(f"  Loading Stage 1 output: {stage1_path}")
    with stage1_path.open() as fh:
        stage1_data: list[dict] = json.load(fh)

    fde_articles = [
        e for e in stage1_data
        if _stage2_mod.is_fde_positive(e)
    ]
    log.info(f"  Total articles in Stage 1 : {len(stage1_data)}")
    log.info(f"  FDE-positive articles     : {len(fde_articles)}")

    if args.limit_s2:
        fde_articles = fde_articles[: args.limit_s2]
        log.info(f"  --limit-s2 active — processing first {args.limit_s2} articles")

    # ── Output file setup ──────────────────────────────────────────────────
    run_ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path      = stage2_dir / f"fde_events_{run_ts}.jsonl"
    csv_path        = stage2_dir / f"fde_events_{run_ts}.csv"
    failed_path     = stage2_dir / f"fde_failed_{run_ts}.jsonl"
    checkpoint_path = stage2_dir / f"fde_checkpoint_{run_ts}.json"

    jsonl_fh  = jsonl_path.open("w",   encoding="utf-8")
    failed_fh = failed_path.open("w",  encoding="utf-8")
    csv_fh    = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=_stage2_mod.CSV_HEADERS)
    csv_writer.writeheader()

    all_results:   list[dict] = []
    pending_batch: list[dict] = []
    stats = {
        "total":        len(fde_articles),
        "processed":    0,
        "events_found": 0,
        "no_events":    0,
        "parse_failed": 0,
        "val_errors":   0,
        "missing_file": 0,
    }

    # ── Batch accumulators ─────────────────────────────────────────────────
    batch_articles: list[str]  = []
    batch_prompts:  list[str]  = []
    batch_metadata: list[dict] = []

    def _flush_batch():
        """Run inference on the accumulated batch and write results."""
        if not batch_articles:
            return

        raw_outputs = _stage2_mod.run_inference(
            pipe           = shared_model.pipe,
            prompts        = batch_prompts,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
        )

        for meta, raw in zip(batch_metadata, raw_outputs):
            article_id = meta["article_id"]
            log.info(f"  Raw output length : {len(raw)} chars")

            parsed = _stage2_mod.extract_json_array(raw)

            if parsed is None:
                log.warning(f"  JSON parse failed for {article_id}")
                failed_fh.write(json.dumps({
                    "article_id": article_id,
                    "error":      "json_parse_failed",
                    "raw_output": raw,
                }) + "\n")
                failed_fh.flush()
                stats["parse_failed"] += 1
                stats["processed"]    += 1
                result_entry = {"article_id": article_id, "status": "json_parse_failed", "raw_output": raw}
                all_results.append(result_entry)
                pending_batch.append(result_entry)
                continue

            if not isinstance(parsed, list) or len(parsed) == 0:
                log.info(f"  {article_id}: no FDE events found by model")
                stats["no_events"]  += 1
                stats["processed"]  += 1
                result_entry = {"article_id": article_id, "status": "no_events", "events": []}
                all_results.append(result_entry)
                pending_batch.append(result_entry)
                continue

            valid_events, invalid_events = _stage2_mod.validate_events(parsed, article_id)

            for ev in invalid_events:
                log.warning(f"  Validation error in {article_id} — logged to failed file")
                failed_fh.write(json.dumps({
                    "article_id": article_id,
                    "error":      "validation_error",
                    "raw_event":  ev,
                }) + "\n")
                failed_fh.flush()
                stats["val_errors"] += 1

            for ev in valid_events:
                jsonl_fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                csv_writer.writerow(_stage2_mod.event_to_csv_row(ev))
            jsonl_fh.flush()
            csv_fh.flush()

            stats["events_found"] += len(valid_events)
            stats["processed"]    += 1
            result_entry = {
                "article_id":    article_id,
                "status":        "ok",
                "valid_events":  valid_events,
                "invalid_events": invalid_events,
            }
            all_results.append(result_entry)
            pending_batch.append(result_entry)
            log.info(
                f"  {article_id}: {len(valid_events)} valid, "
                f"{len(invalid_events)} invalid  "
                f"(running total: {stats['events_found']})"
            )

        # Periodic checkpoint
        if len(pending_batch) >= args.checkpoint_every:
            _stage2_mod._atomic_json_dump(checkpoint_path, all_results)
            log.info(f"  Checkpoint saved → {checkpoint_path.name}")
            pending_batch.clear()

        batch_articles.clear()
        batch_prompts.clear()
        batch_metadata.clear()

    # ── Main loop ──────────────────────────────────────────────────────────
    for idx, entry in enumerate(fde_articles, 1):
        article_id = entry["article_id"]

        # Stage 0 saves files as e<DOC_ID>.json; Stage 1 uses the stem as
        # article_id.  Both conventions map to the same file name.
        text_file = text_dir / f"{article_id}.json"

        log.info(f"[{idx}/{stats['total']}] {article_id}")

        if not text_file.exists():
            log.warning(f"  Article file not found: {text_file} — skipping")
            failed_fh.write(json.dumps({
                "article_id": article_id,
                "error": "file_not_found",
                "path":  str(text_file),
            }) + "\n")
            failed_fh.flush()
            stats["missing_file"] += 1
            continue

        article_text = text_file.read_text(encoding="utf-8", errors="replace").strip()

        prompt = _stage2_mod.build_prompt(
            tokenizer  = shared_model.tokenizer,
            article_id = article_id,
            article_text = article_text,
        )
        batch_articles.append(article_id)
        batch_prompts.append(prompt)
        batch_metadata.append({"article_id": article_id})

        if len(batch_articles) >= args.batch_size or idx == stats["total"]:
            _flush_batch()

    # ── Close files ────────────────────────────────────────────────────────
    jsonl_fh.close()
    failed_fh.close()
    csv_fh.close()

    if all_results:
        _stage2_mod._atomic_json_dump(checkpoint_path, all_results)

    log.info(f"Stage 2 complete  ({time.time()-t0:.1f}s)")
    log.info(f"  Articles processed       : {stats['processed']} / {stats['total']}")
    log.info(f"  Valid FDE events         : {stats['events_found']}")
    log.info(f"  Articles → no events     : {stats['no_events']}")
    log.info(f"  JSON parse failures      : {stats['parse_failed']}")
    log.info(f"  Pydantic validation errs : {stats['val_errors']}")
    log.info(f"  Missing article files    : {stats['missing_file']}")
    log.info(f"  Output JSONL             : {jsonl_path}")
    log.info(f"  Output CSV               : {csv_path}")
    log.info(f"  Failed log               : {failed_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fisheries Conflict Pipeline — runs Stage 0 → 1 → 2 with a "
            "single LLM load."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input (Stage 0) ────────────────────────────────────────────────────
    inp = p.add_mutually_exclusive_group()
    inp.add_argument("--zip",    type=Path, metavar="PATH",
                     help="ZIP archive of LexisNexis article files (.docx)")
    inp.add_argument("--folder", type=Path, metavar="PATH",
                     help="Folder containing article files (.docx / .doc / .txt)")
    inp.add_argument("--file",   type=Path, metavar="PATH",
                     help="Single article file (.docx / .doc / .txt)")
    inp.add_argument("--string", type=str, metavar="TEXT",
                     help=(
                         "Single raw article string passed inline. "
                         "Quote the whole string on the CLI. "
                         "The doc_id is DOC_<id-start> (default DOC_0001)."
                     ))
    inp.add_argument("--strings-json", type=Path, metavar="PATH",
                     help=(
                         "JSON file containing a batch of raw article strings. "
                         "Accepted formats:\n"
                         '- plain array:   ["text1", "text2", ...]\n'
                         '- object array:  [{"doc_id": "X", "text": "..."}, ...]\n'
                         "Objects may omit doc_id; position index is used as fallback."
                     ))

    # ── Output ─────────────────────────────────────────────────────────────
    p.add_argument("--output", type=Path, default=Path("results"),
                   help="Root output directory (default: results/)")

    # ── Model ──────────────────────────────────────────────────────────────
    p.add_argument("--model",  default="google/gemma-4-E4B-it",
                   help="HuggingFace model ID (default: google/gemma-4-E4B-it)")
    p.add_argument("--device", default="auto",
                   help="Device: auto | cpu | cuda | cuda:0  (default: auto)")
    p.add_argument("--device-map-mode", choices=("single", "auto"), default="single",
                   help="'single' for one GPU, 'auto' to shard across multiple GPUs")

    # ── Stage-skip flags ───────────────────────────────────────────────────
    p.add_argument("--skip-stage0", action="store_true",
                   help="Skip text extraction; reuse files already in <output>/stage0/")
    p.add_argument("--skip-stage1", action="store_true",
                   help="Skip dispute detection; requires --stage1-file")
    p.add_argument("--stage1-file", type=Path, default=None,
                   help="Path to existing Stage 1 JSON output (required with --skip-stage1)")

    # ── Inference tuning ───────────────────────────────────────────────────
    p.add_argument("--max-new-tokens",  type=int,   default=2048)
    p.add_argument("--temperature",     type=float, default=0.1)
    p.add_argument("--checkpoint-every",type=int,   default=50,
                   help="Checkpoint frequency in articles (default: 50)")
    p.add_argument("--batch-size",      type=int,   default=4,
                   help="Inference batch size for Stage 2 (default: 4)")

    # ── Limits ────────────────────────────────────────────────────────────
    p.add_argument("--id-start",  type=int, default=1,
                   help="Starting DOC_XXXX id for Stage 0 (default: 1)")
    p.add_argument("--limit-s1",  type=int, default=None,
                   help="Process only the first N articles in Stage 1 (testing)")
    p.add_argument("--limit-s2",  type=int, default=None,
                   help="Process only the first N FDE-positive articles in Stage 2 (testing)")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ── Validate skip flags ────────────────────────────────────────────────
    if args.skip_stage1 and args.stage1_file is None:
        log.error("--skip-stage1 requires --stage1-file <path>")
        sys.exit(1)
 
    if not args.skip_stage0 and not any([
        args.zip, args.folder, args.file, args.string, args.strings_json
    ]):
        log.error(
            "Provide an input source for Stage 0: "
            "--zip / --folder / --file / --string / --strings-json "
            "(or use --skip-stage0 to reuse existing extracted files)"
        )
        sys.exit(1)

    # ── Directory layout ───────────────────────────────────────────────────
    output_dir  = args.output
    text_dir    = output_dir / "stage0"
    stage1_dir  = output_dir / "stage1"
    stage2_dir  = output_dir / "stage2"

    log.info("=" * 60)
    log.info("FISHERIES DISPUTE EVENT PIPELINE")
    log.info("=" * 60)
    log.info(f"  Output root  : {output_dir.resolve()}")
    log.info(f"  Model        : {args.model}")
    log.info(f"  Device       : {args.device}")
    log.info(f"  Skip Stage 0 : {args.skip_stage0}")
    log.info(f"  Skip Stage 1 : {args.skip_stage1}")

    # ── Stage 0 ───────────────────────────────────────────────────────────
    if args.skip_stage0:
        log.info("Stage 0 skipped — using existing files in stage0/")
    else:
        run_stage0(args, text_dir)

    # ── Load model (once, before Stages 1 & 2) ────────────────────────────
    if not args.skip_stage1 or True:   # always need model for Stage 2
        log.info("=" * 60)
        log.info("Loading LLM (single load for all remaining stages) …")
        log.info("=" * 60)
        shared_model = load_shared_model(
            model_name      = args.model,
            device          = args.device,
            device_map_mode = args.device_map_mode,
            stage2_batch_size = args.batch_size,
        )
    else:
        shared_model = None

    # ── Stage 1 ───────────────────────────────────────────────────────────
    if args.skip_stage1:
        stage1_path = args.stage1_file
        log.info(f"Stage 1 skipped — using: {stage1_path}")
    else:
        stage1_path = run_stage1(args, text_dir, stage1_dir, shared_model)

    # ── Stage 2 ───────────────────────────────────────────────────────────
    run_stage2(args, stage1_path, text_dir, stage2_dir, shared_model)

    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info(f"  Results in: {output_dir.resolve()}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
