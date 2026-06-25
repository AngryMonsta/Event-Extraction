"""Stage1 dispute-detection.

Example:
    python dispute_detection.py \
        --data-folder proto_data/ \
        --extraction-folder TextExtracted \
        --output-filename Gemma4_e4b_summary \
        --model-id google/gemma-4-E4B-it \
        --checkpoint-every 3 \
        --limit 10
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import torch
# from accelerate import Accelerator
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
# from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


class DetectionSchema(BaseModel):
    """Stage 1 output schema - article-level FDE / FDA detection."""

    contains_fde: bool = Field(
        description="True if the article contains at least one Fisheries Dispute Event (FDE)."
    )
    contains_fda: bool = Field(
        description="True if the article contains at least one Fisheries Dispute Aggregate (FDA)."
    )
    num_candidate_events: int = Field(
        description="Total number of FDE and FDA found in the article."
    )
    confidence: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Confidence in the article-level classification (0-1).",
    )
    evidence_summary: str = Field(
        description="Brief evidence-based explanation citing relevant article excerpts."
    )


class OutputFileSchema(BaseModel):
    article_id: str
    detection: List[DetectionSchema]


def _extract_json_from_text(text: str) -> str:
    """Strip markdown fences and extract the final JSON object or array."""

    text = text.strip()

    for pattern in (r"```json(.*?)```", r"```(.*?)```"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    last_obj = text.rfind("{")
    last_arr = text.rfind("[")
    start = max(last_obj, last_arr)

    if start == -1:
        return text

    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    depth = 0
    for index in range(start, len(text)):
        if text[index] == open_char:
            depth += 1
        elif text[index] == close_char:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return text[start:]


def _repair_truncated_json(text: str) -> str:
    """Best-effort close a truncated JSON object or array."""

    stack = []
    in_string = False
    escaped = False

    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char in ("}", "]") and stack and stack[-1] == char:
                stack.pop()

    repaired = text.rstrip()

    if in_string:
        repaired += '"'

    while stack:
        repaired += stack.pop()

    return repaired


def _atomic_json_dump(path: Path, payload) -> None:
    """Write JSON atomically so partial checkpoints are not corrupted."""

    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


def load_model(model_name: str, device: str = "auto", device_map_mode: str = "single"):
    """Load the processor and causal LM used for detection."""

    start_time = time.time()
    print(f"Loading processor : {model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"Processor ready   : {time.time() - start_time:.1f}s", flush=True)

    if device_map_mode == "auto":
        device_map = "auto"
    else:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_map = {"": device}

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model     : {model_name}  (dtype={dtype}, device_map={device_map})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Model ready      : {time.time() - start_time:.1f}s", flush=True)
    print("Device:", next(model.parameters()).device, flush=True)

    return processor, model


def build_system_prompt(schema_json: str) -> str:
    return textwrap.dedent(
        f"""
        You are a fisheries conflict event code engine.
        Analyse the supplied article and return a single JSON object.

        Definitions:

          Fisheries Dispute Events (FDE):
            An incident in which a fisheries resource is contested, disputed, or the source
            of conflict between a minimum of two actors, at a discrete temporal moment and
            in a discrete location.

          Fisheries Dispute Aggregates (FDA):
            An incident or incidents where a fisheries resource is contested, disputed,
            or the source of conflict between a minimum of two actors, over an uncertain
            or prolonged period of time at a discrete waterbody-district geographic scale.

          Actors:
            human individuals or groups (spontaneous or organized)
            Domestic Fishers, Other, Foreign Fishers, Federal Government, Bandits,
            Military, Police, Regional Government, Resource Security, Local Government.

        Task: Analyse the article at the ARTICLE level.

          1. Count total fisheries conflict events -> num_candidate_events.
          2. Set contains_fde = true if >=1 FDE is present.
          3. Set contains_fda = true if >=1 FDA is present.
          4. Provide a brief evidence-based summary (<=100 words) citing the strongest evidence.
          5. Return ONLY this JSON object. No preamble, no explanation, no markdown fences.
            {schema_json}

        Rules:
          - Use only information stated in the article.
          - Do not infer missing facts.
          - Output JSON only.
        """
    ).strip()


def run_detection(
    extracted_content: Dict,
    processor,
    model,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> DetectionSchema | Dict:
    """Run article-level dispute detection on one extracted article."""

    article_text = extracted_content.get("article_text", "")
    doc_id = extracted_content.get("doc_id", "N/A")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Article text: {article_text}"},
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # If the context + generation would be very large, treat this sample specially.
    safe_total = int(os.getenv("MAX_TOTAL_TOKENS", "8192"))
    total_tokens = input_len + max_new_tokens
    if total_tokens > safe_total:
        # Per-request override: double the requested generation budget for this sample
        # as requested. If this fails (OOM), skip this sample instead of retrying.
        attempt_tokens = safe_total + max_new_tokens
        print(f"  WARNING: sample total_tokens={total_tokens} > safe_total={safe_total}; trying increased max_new_tokens={attempt_tokens}", flush=True)
    else:
        attempt_tokens = max_new_tokens

    output_ids = None
    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=attempt_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
    except RuntimeError as runtime_err:
        msg = str(runtime_err).lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            print(f"  [OOM] generation failed for doc {doc_id} with attempt_tokens={attempt_tokens}: {runtime_err}", flush=True)
            # free cache and skip this sample
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()
            return {"error": "SKIPPED_OOM", "details": str(runtime_err)}
        raise

    raw_response = processor.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    ).strip()

    # _ = processor.parse_response(raw_response)
    extracted_json = _extract_json_from_text(raw_response)

    try:
        detection_data = json.loads(extracted_json)
        return TypeAdapter(DetectionSchema).validate_python(detection_data)
    except json.JSONDecodeError as error:
        repaired_json = _repair_truncated_json(extracted_json)
        if repaired_json != extracted_json:
            try:
                detection_data = json.loads(repaired_json)
                return TypeAdapter(DetectionSchema).validate_python(detection_data)
            except (json.JSONDecodeError, ValidationError) as repair_error:
                print(
                    f"  [JSON ERROR] {doc_id}: {error}\n"
                    f"  Repair attempt failed: {repair_error}\n"
                    f"  Raw prefix: {repaired_json[:300]}"
                )
                return {
                    "error": "JSON_DECODE_ERROR",
                    "details": str(error),
                    "repaired_details": str(repair_error),
                    "raw_output": raw_response,
                }

        print(f"  [JSON ERROR] {doc_id}: {error}\n  Raw prefix: {extracted_json[:300]}")
        return {"error": "JSON_DECODE_ERROR", "details": str(error), "raw_output": raw_response}
    except ValidationError as error:
        print(f"  [VALIDATION ERROR] {doc_id}: {error}\n  Raw prefix: {extracted_json[:300]}")
        return {"error": "VALIDATION_ERROR", "details": str(error), "raw_output": raw_response}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dispute detection over eDOC_*.json files.")
    parser.add_argument(
        "--data-folder",
        type=Path,
        required=True,
        help="Root data folder containing TextExtracted/ and the output folder.",
    )
    parser.add_argument(
        "--extraction-folder",
        type=str,
        default="TextExtracted",
        help="Subfolder containing eDOC_*.json files, relative to --data-folder.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default="DisputeDetect",
        help="Subfolder where the aggregated output will be written.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="Gemma4_e4b_summary",
        help="Output JSON filename without .json.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-4-E4B-it",
        help="Hugging Face model identifier.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate per article.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Write the incremental checkpoint after this many newly processed samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N articles.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device or device map passed to transformers/Accelerate.",
    )
    parser.add_argument(
        "--device-map-mode",
        type=str,
        choices=("single", "auto"),
        default="single",
        help="Use 'single' for one visible GPU or 'auto' to shard across multiple GPUs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_folder = args.data_folder.expanduser().resolve()
    extraction_folder_path = data_folder / args.extraction_folder
    outputs_folder_path = data_folder / args.output_folder
    outputs_folder_path.mkdir(parents=True, exist_ok=True)

    if not extraction_folder_path.is_dir():
        raise FileNotFoundError(f"Extraction folder not found: {extraction_folder_path}")

    extraction_files = sorted(
        file_name for file_name in os.listdir(extraction_folder_path) if file_name.startswith("eDOC_")
    )

    if args.limit is not None:
        extraction_files = extraction_files[: args.limit]
        print(f"LIMIT active - processing first {args.limit} files")

    summary_file_path = outputs_folder_path / f"{args.output_filename}_{args.extraction_folder}.json"
    checkpoint_file_path = outputs_folder_path / f"{args.output_filename}_{args.extraction_folder}.checkpoint.json"

    print(f"Working directory: {os.getcwd()}")
    print(f"Extraction folder : {extraction_folder_path}")
    print(f"Output folder     : {outputs_folder_path}")
    print(f"Output file       : {summary_file_path}")
    print(f"Checkpoint file   : {checkpoint_file_path}")
    print(f"Articles to process: {len(extraction_files)}")

    detection_schema_json = json.dumps(TypeAdapter(DetectionSchema).json_schema(), indent=2)
    system_prompt = build_system_prompt(detection_schema_json)

    processor, model = load_model(args.model_id, device=args.device, device_map_mode=args.device_map_mode)

    all_dispute_detections = []
    error_count = 0
    processed_ids = set()
    processed_count = 0
    pending_batch = []

    if checkpoint_file_path.exists():
        with checkpoint_file_path.open("r", encoding="utf-8") as handle:
            checkpoint_entries = json.load(handle)

        if isinstance(checkpoint_entries, list):
            all_dispute_detections = checkpoint_entries
            processed_ids = {
                entry.get("article_id")
                for entry in checkpoint_entries
                if isinstance(entry, dict) and entry.get("article_id")
            }
            processed_count = len(all_dispute_detections)
            error_count = sum(1 for entry in checkpoint_entries if isinstance(entry, dict) and "detection_error" in entry)
            print(f"Resuming from checkpoint with {len(all_dispute_detections)} saved entries")
        else:
            raise TypeError(f"Expected list in checkpoint file {checkpoint_file_path}, got {type(checkpoint_entries)}")

    for file_name in extraction_files:
        article_id = file_name.removesuffix(".json")
        if article_id in processed_ids:
            print(f"  {article_id}: skipped (already checkpointed)")
            continue

        file_path = extraction_folder_path / file_name

        with file_path.open("r", encoding="utf-8") as handle:
            extracted_content = json.load(handle)

        if not isinstance(extracted_content, dict):
            raise TypeError(f"Expected dict in {file_path}, got {type(extracted_content)}")

        result = run_detection(
            extracted_content,
            processor,
            model,
            system_prompt,
            args.max_new_tokens,
            args.temperature,
        )

        if isinstance(result, DetectionSchema):
            entry = OutputFileSchema(article_id=article_id, detection=[result]).model_dump(mode="json")
            status = (
                f"fde={result.contains_fde}  fda={result.contains_fda}  "
                f"n={result.num_candidate_events}  conf={result.confidence}"
            )
        else:
            entry = {"article_id": article_id, "detection_error": result}
            status = f"ERROR - {result.get('error')}"
            error_count += 1

        all_dispute_detections.append(entry)
        pending_batch.append(entry)
        processed_ids.add(article_id)
        processed_count += 1
        print(f"  {article_id}: {status}")

        if len(pending_batch) >= args.checkpoint_every:
            _atomic_json_dump(checkpoint_file_path, all_dispute_detections)
            print(
                f"  checkpoint saved -> {checkpoint_file_path} "
                f"({len(all_dispute_detections)} entries, batch={len(pending_batch)})"
            )
            pending_batch.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\nDone. {len(all_dispute_detections)} articles processed, {error_count} errors.")

    if pending_batch:
        _atomic_json_dump(checkpoint_file_path, all_dispute_detections)
        print(
            f"  final checkpoint saved -> {checkpoint_file_path} "
            f"({len(all_dispute_detections)} entries, batch={len(pending_batch)})"
        )

    with summary_file_path.open("w", encoding="utf-8") as out:
        json.dump(all_dispute_detections, out, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_dispute_detections)} entries -> {summary_file_path}")

    successes = [entry for entry in all_dispute_detections if "detection" in entry]
    errors = [entry for entry in all_dispute_detections if "detection_error" in entry]
    fde_hits = sum(1 for entry in successes if entry["detection"][0]["contains_fde"])
    fda_hits = sum(1 for entry in successes if entry["detection"][0]["contains_fda"])

    print(f"Total articles : {len(all_dispute_detections)}")
    print(f"Successful     : {len(successes)}")
    print(f"Errors         : {len(errors)}")
    print(f"contains_fde=T : {fde_hits}")
    print(f"contains_fda=T : {fda_hits}")

    if successes:
        print("\nFirst successful result:")
        print(json.dumps(successes[0], indent=2))


if __name__ == "__main__":
    main()