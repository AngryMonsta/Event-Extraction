"""
Stage 2: FDE Extraction Pipeline — Local LLM (Gemma 4, HuggingFace Transformers)
==================================================================================
Reads Stage 1 output summary JSON, filters articles where contains_fde=True,
loads article text from a separate folder, feeds each article to a local
Gemma 4 model, validates the output against the FdeEvent Pydantic schema,
and writes both JSONL and CSV outputs.

Directory layout expected:
    stage1_output.json          ← Stage 1 detection results
    articles/
        nDOC_001_ext.json
        nDOC_002_ext.json
        ...

Usage:
    python fde_local_pipeline.py \\
        --stage1    ../proto_data/Gemma4_e4b_summary_2.json \\
        --articles  ../proto_data/n_extracted/ \\
        --output    ../results/ \\
        --model     google/gemma-4-E4B-it

    # Limit to N articles (for testing)
    python fde_local_pipeline.py \\
        --stage1 ../proto_data/Gemma4_e4b_summary_2.json \\
        --articles ../proto_data/n_extracted/ \\
        --limit 5
"""

import argparse
import csv
import json
import logging
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from pydantic import BaseModel, Field, ValidationError
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

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
# Pydantic schema (exactly as specified)
# ---------------------------------------------------------------------------

class FdeCriteria(BaseModel):
    min_two_actors: bool = Field(description="At least two actors are involved")
    discrete_temporal_moment: bool = Field(
        description="Event has a discrete time period (not merely systemic)"
    )
    discrete_location: bool = Field(
        description="Event has a bounded, identifiable location"
    )
    fisheries_resource_contested: bool = Field(
        description="A fisheries resource is contested/disputed/source of conflict"
    )


class ConflictDrivers(BaseModel):
    WeakGov: bool
    FishPop: bool
    EcoChngOther: bool
    GroundsLim: bool
    ForeignFisher: bool
    IllegalFishing: bool
    IncrEfficiency: bool
    OpsScales: bool
    Markets: bool
    FoodInsecurity: bool
    Marginalization: bool
    StratLoc: bool
    MaritimeCrime: bool


class FdeEvent(BaseModel):
    article_id: str
    fde_criteria: FdeCriteria
    date: str = Field(description="Date FDE began in YYYY-MM-DD. Use best estimate if uncertain.")
    year: int = Field(description="Year of event")
    country: str = Field(description="Country in which the FDE occurred")
    region: Literal[0, 1, 2, 3, 4] = Field(
        description="0=Arctic 1=Horn_of_Africa 2=Oceania 3=South_America 4=South_Asia"
    )
    location: str = Field(description="Closest identifiable land-based location")
    loc_accuracy: Literal[0, 1, 2, 3, 4, 5] = Field(
        description="0=Country 1=District 2=International 3=NA 4=Region 5=Town"
    )
    actor1: str = Field(description="Primary actor")
    actor2: str = Field(description="Secondary actor")
    ViolenceScore: int = Field(
        ge=1, le=3,
        description=(
            "1=Verbal/complaints/bans/fines  "
            "2=Property_damage/confiscation/arrests/abductions  "
            "3=Injuries/assaults/fatalities"
        ),
    )
    drivers: ConflictDrivers
    issue_note: str = Field(
        description=(
            "1-2 sentences summarising the event AND justifying key coding choices "
            "(ViolenceScore, actors, any true drivers)."
        )
    )


# ---------------------------------------------------------------------------
# Codebook system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert coder for the Fisheries Conflict Database.
Read the supplied news article and extract every distinct Fisheries Dispute
Event (FDE) it contains. Return ONLY a valid JSON array — no markdown fences,
no commentary. If no FDEs are found, return [].

━━━ DEFINITIONS ━━━
FDE: An incident in which a fisheries resource is contested, disputed, or the
source of conflict between ≥2 ACTORS, at a DISCRETE TEMPORAL MOMENT, in a
DISCRETE LOCATION.
• Actor: a human individual or group (both sides must be human).
• Discrete temporal moment: days/weeks, NOT a chronic systemic condition.
• Discrete location: a bounded site (village, reef, landing), NOT a diffuse region.

━━━ REGION CODES ━━━
0=Arctic
1=Horn of Africa (Tanzania, Kenya, Somalia, Djibouti, Eritrea, Yemen, Ethiopia,
  Sudan, Uganda, Rwanda, Burundi, DRC, Lake Victoria basin, Red Sea,
  Indian Ocean off East Africa)
2=Oceania
3=South America
4=South Asia

━━━ LOC_ACCURACY CODES ━━━
0=Country only  1=District/Province  2=International/High-seas
3=NA (indeterminate)  4=Region (named but no district/town)  5=Town/village/named site

━━━ VIOLENCE SCORE ━━━
1 = Verbal only — bans, complaints, fines, policy disputes
2 = Physical action, NO bodily harm — property damage, gear/boat confiscation,
    arrests, abductions
3 = Bodily harm — injuries, sexual assault, fatalities
Assign the HIGHEST level met for the event.

━━━ CONFLICT DRIVERS (true ONLY if explicitly supported by article text) ━━━
WeakGov        Corruption, weak enforcement, institutional capacity gaps
FishPop        Actual/perceived decline in fish stocks motivating actors
EcoChngOther   Non-fish ecosystem change (pollution, coral damage, acidification)
Poverty        Poverty (not food insecurity) motivating actors
FoodInsecurity Lack of access to reliable nutritious food motivating actors
Marginalization Social/ethnic/gender/political marginalization
GroundsLim     Limits on access to fishing grounds (borders, parks, EEZ, closed areas)
OpsScales      Competition between industrial/commercial vs artisanal/subsistence fleets
ForeignFisher  Presence of foreign fishers in domestic waters driving actors
Markets        Transnational market supply/demand driving actors
IncrEfficiency Increased fishing efficiency (sonar, finer nets, destructive gear)
IncrPressure   Increased fishing pressure (population growth, fleet expansion)
Crime          Non-fisheries maritime crime: piracy, smuggling, trafficking, ransom
StratLoc       Strategic importance of land location (not the fish itself)

━━━ CODING RULES ━━━
1. Evidence standard: a driver is true if the article states it OR if the
   article contains context that directly implies it (e.g. "fishing unabated
   for five years despite ban" implies WeakGov without using that word).
2. Each distinct event (different date OR location OR actors) = a separate FDE.
3. Aggregate/trend text without a discrete event → no FDE.
4. date: YYYY-MM-DD; use -01 placeholders for unknown month/day.
5. actor1 = initiating/primary party; actor2 = responding/secondary party.
6. issue_note: for EVERY true driver, quote or paraphrase the article phrase
   that triggered it. This is mandatory — do not assert a driver without
   citing evidence from the text.
7. Return only the JSON array — no extra text before or after.

━━━ OUTPUT SCHEMA (one object per FDE) ━━━
{
  "article_id": "<string>",
  "fde_criteria": {
    "min_two_actors": <bool>,
    "discrete_temporal_moment": <bool>,
    "discrete_location": <bool>,
    "fisheries_resource_contested": <bool>
  },
  "date": "<YYYY-MM-DD>",
  "year": <int>,
  "country": "<string>",
  "region": <0|1|2|3|4>,
  "location": "<string>",
  "loc_accuracy": <0|1|2|3|4|5>,
  "actor1": "<string>",
  "actor2": "<string>",
  "ViolenceScore": <1|2|3>,
  "drivers": {
    "WeakGov": <bool>, "FishPop": <bool>, "EcoChngOther": <bool>, 
    "GroundsLim": <bool>, "ForeignFisher": <bool>, "IllegalFishing": <bool>, 
    "IncrEfficiency": <bool>, "OpsScales": <bool>, "Markets": <bool>,
    "FoodInsecurity": <bool>, "Marginalization": <bool>, "StratLoc": <bool>, 
    "MaritimeCrime": <bool>
  },
  "issue_note": "<string>"
}
""").strip()


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str) -> tuple:
    """
    Load a Gemma 4 (or any causal LM) model and tokenizer from HuggingFace.
    Returns (pipe) — a transformers text-generation pipeline.
    """
    log.info(f"Loading tokenizer: {model_name}")
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    log.info(f"Loading model: {model_name}  (device={device})")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "auto" else device,
        low_cpu_mem_usage=True,
    )
    model = accelerator.prepare(model)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if device == "auto" else None,
    )
    log.info("Model loaded successfully.")
    return pipe


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, article_id: str, article_text: str) -> str:
    """
    Build a chat-formatted prompt using the tokenizer's chat template
    (works for Gemma-IT, Llama-Instruct, Qwen-Chat, etc.).
    Falls back to a plain [INST]-style prompt if no chat template is defined.
    """

    user_content = (
        f"ARTICLE ID: {article_id}\n\n"
        f"{article_text.strip()}\n\n"
        "Extract all FDE events from this article and return them as a JSON array."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    # Gemma-IT and most modern instruct models have apply_chat_template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            return prompt
        except Exception:
            pass  # fall through to manual format

    # Fallback: simple instruction wrapper
    return (
        f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
        f"<start_of_turn>user\n{user_content}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    pipe,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
) -> str:
    """Run the model and return only the newly generated text."""
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,   # only the generated part, not the prompt
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    return outputs[0]["generated_text"].strip()


# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------

def extract_json_array(raw: str) -> Optional[list]:
    """
    Pull a JSON array out of the model's raw output.
    Handles: clean JSON, markdown-fenced JSON, JSON embedded in prose.
    Returns the parsed list or None on failure.
    """
    # 1. Strip markdown fences
    text = raw
    if "```" in text:
        parts = text.split("```")
        # Take the content inside the first fence block
        for part in parts[1::2]:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # 2. Try the raw text directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 3. Find the first '[' and last ']' and try that substring
    start = text.find("[")
    end   = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def validate_events(raw_events: list, article_id: str) -> tuple[list[dict], list[dict]]:
    """
    Validate each raw dict through FdeEvent Pydantic schema.
    Returns (valid_records, invalid_records).
    invalid_records include the raw dict + '_validation_error' key.
    """
    valid, invalid = [], []
    for i, ev in enumerate(raw_events):
        # Inject article_id if model forgot it
        if "article_id" not in ev or not ev["article_id"]:
            ev["article_id"] = article_id
        try:
            fde = FdeEvent(**ev)
            valid.append(fde.model_dump())
        except ValidationError as e:
            ev["_validation_error"] = str(e)
            ev["_source_article_id"] = article_id
            invalid.append(ev)
    return valid, invalid


# ---------------------------------------------------------------------------
# Flat CSV row builder
# ---------------------------------------------------------------------------

# All driver field names in order
DRIVER_FIELDS = [
    "WeakGov", "FishPop", "EcoChngOther", "GroundsLim", "ForeignFisher",
    "IllegalFishing", "IncrEfficiency", "OpsScales", "Markets",
    "FoodInsecurity", "Marginalization", "StratLoc", "MaritimeCrime"
]

CSV_HEADERS = [
    "article_id", "date", "year", "country", "region",
    "location", "loc_accuracy", "actor1", "actor2", "ViolenceScore",
    "fde_min_two_actors", "fde_discrete_temporal_moment",
    "fde_discrete_location", "fde_fisheries_resource_contested",
] + DRIVER_FIELDS + ["issue_note"]


def event_to_csv_row(ev: dict) -> dict:
    crit = ev.get("fde_criteria", {})
    drv  = ev.get("drivers", {})
    return {
        "article_id":    ev.get("article_id", ""),
        "date":          ev.get("date", ""),
        "year":          ev.get("year", ""),
        "country":       ev.get("country", ""),
        "region":        ev.get("region", ""),
        "location":      ev.get("location", ""),
        "loc_accuracy":  ev.get("loc_accuracy", ""),
        "actor1":        ev.get("actor1", ""),
        "actor2":        ev.get("actor2", ""),
        "ViolenceScore": ev.get("ViolenceScore", ""),
        # "fde_min_two_actors":              int(crit.get("min_two_actors", False)),
        # "fde_discrete_temporal_moment":    int(crit.get("discrete_temporal_moment", False)),
        # "fde_discrete_location":           int(crit.get("discrete_location", False)),
        # "fde_fisheries_resource_contested":int(crit.get("fisheries_resource_contested", False)),
        **{k: int(drv.get(k, False)) for k in DRIVER_FIELDS},
        "issue_note": ev.get("issue_note", ""),
    }

# ---------------------------------------------------------------------------
# Stage 1 normaliser
# ---------------------------------------------------------------------------
 
def get_detections(entry: dict) -> list[dict]:
    """
    Return the detections list from a Stage 1 entry regardless of whether
    the key is spelled 'detection' or 'detected' (both appear in the wild).
    Logs a warning if neither key is present.
    """
    if "detection" in entry:
        return entry["detection"] or []
    if "detected" in entry:
        return entry["detected"] or []
    log.warning(
        f"  Article {entry.get('article_id', '?')} has neither "
        f"'detection' nor 'detected' key — skipping."
    )
    return []
 
 
def is_fde_positive(entry: dict) -> bool:
    """Return True if ANY detection record has contains_fde=True."""
    return any(d.get("contains_fde", False) for d in get_detections(entry))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 FDE Extraction — Local Gemma 4 (HuggingFace Transformers)"
    )
    parser.add_argument("--stage1",   required=True,
                        help="Path to Stage 1 output JSON file")
    parser.add_argument("--articles", required=True,
                        help="Path to folder containing article .txt files")
    parser.add_argument("--output",   default="../results/",
                        help="Output directory (default: ../results/)")
    parser.add_argument("--model",    default="google/gemma-4-E4B-it",
                        help="HuggingFace model ID (default: google/gemma-4-E4B-it)")
    parser.add_argument("--device",   default="auto",
                        help="Device: auto | cpu | cuda | cuda:0 (default: auto)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max tokens to generate per article (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1 — near-deterministic)")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Process only the first N FDE-positive articles (for testing)")
    args = parser.parse_args()

    # ── Paths ──────────────────────────────────────────────────────────────
    stage1_path   = Path(args.stage1)
    articles_dir  = Path(args.articles)
    output_dir    = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path   = output_dir / f"fde_events_{run_ts}.jsonl"
    csv_path     = output_dir / f"fde_events_{run_ts}.csv"
    failed_path  = output_dir / f"fde_failed_{run_ts}.jsonl"

    # ── Load Stage 1 output ────────────────────────────────────────────────
    log.info(f"Loading Stage 1 output: {stage1_path}")
    with open(stage1_path) as f:
        stage1_data: list[dict] = json.load(f)

    # Filter to articles where ANY detected entry has contains_fde=True
    fde_articles = [
        entry for entry in stage1_data
        if is_fde_positive(entry)
    ]
    log.info(
        f"  Total articles in Stage 1: {len(stage1_data)}"
    )
    log.info(
        f"  Articles with contains_fde=True: {len(fde_articles)}"
    )

    if args.limit:
        fde_articles = fde_articles[: args.limit]
        log.info(f"  Limiting to first {args.limit} articles (--limit flag)")

    # ── Load model ─────────────────────────────────────────────────────────
    pipe = load_model(args.model, args.device)

    # ── Open output files ──────────────────────────────────────────────────
    jsonl_fh  = open(jsonl_path,  "w", encoding="utf-8")
    failed_fh = open(failed_path, "w", encoding="utf-8")
    csv_fh    = open(csv_path,    "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=CSV_HEADERS)
    csv_writer.writeheader()

    # ── Per-article stats ──────────────────────────────────────────────────
    stats = {
        "total":        len(fde_articles),
        "processed":    0,
        "events_found": 0,
        "no_events":    0,
        "parse_failed": 0,
        "val_errors":   0,
        "missing_file": 0,
    }

    # ── Main loop ──────────────────────────────────────────────────────────
    for idx, entry in enumerate(fde_articles, 1):
        article_id = entry["article_id"]
        text_file   = articles_dir / f"{article_id}_ext.json"

        log.info(f"[{idx}/{stats['total']}] {article_id}")

        # ── Load article text ──────────────────────────────────────────────
        if not text_file.exists():
            log.warning(f"  Article file not found: {text_file} — skipping")
            failed_fh.write(json.dumps({
                "article_id": article_id,
                "error": "file_not_found",
                "path": str(text_file),
            }) + "\n")
            stats["missing_file"] += 1
            continue

        article_text = text_file.read_text(encoding="utf-8", errors="replace").strip()

        # ── Build prompt & run inference ───────────────────────────────────
        t0     = time.time()
        prompt = build_prompt(pipe.tokenizer, article_id, article_text)
        raw    = run_inference(
            pipe, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        elapsed = time.time() - t0
        log.info(f"  Inference: {elapsed:.1f}s  |  Raw output length: {len(raw)} chars")

        # ── Parse JSON ─────────────────────────────────────────────────────
        parsed = extract_json_array(raw)

        if parsed is None:
            log.warning(f"  JSON parse failed — logging to {failed_path.name}")
            failed_fh.write(json.dumps({
                "article_id": article_id,
                "error":      "json_parse_failed",
                "raw_output": raw,
            }) + "\n")
            stats["parse_failed"] += 1
            stats["processed"] += 1
            continue

        if not isinstance(parsed, list) or len(parsed) == 0:
            log.info(f"  Model returned empty array (no FDEs found in article)")
            stats["no_events"] += 1
            stats["processed"] += 1
            continue

        # ── Validate ───────────────────────────────────────────────────────
        valid_events, invalid_events = validate_events(parsed, article_id)

        for ev in invalid_events:
            log.warning(f"  Validation error on 1 event — logged to failed file")
            failed_fh.write(json.dumps({
                "article_id": article_id,
                "error":      "validation_error",
                "raw_event":  ev,
            }) + "\n")
            stats["val_errors"] += 1

        # ── Write valid events ─────────────────────────────────────────────
        for ev in valid_events:
            jsonl_fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            csv_writer.writerow(event_to_csv_row(ev))

        stats["events_found"] += len(valid_events)
        stats["processed"]    += 1

        log.info(
            f"  Valid events: {len(valid_events)}  |  "
            f"Invalid: {len(invalid_events)}  |  "
            f"Running total: {stats['events_found']}"
        )

    # ── Close files ────────────────────────────────────────────────────────
    jsonl_fh.close()
    failed_fh.close()
    csv_fh.close()

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info(f"  Articles processed       : {stats['processed']} / {stats['total']}")
    log.info(f"  Valid FDE events found   : {stats['events_found']}")
    log.info(f"  Articles → no events     : {stats['no_events']}")
    log.info(f"  JSON parse failures      : {stats['parse_failed']}")
    log.info(f"  Pydantic validation errs : {stats['val_errors']}")
    log.info(f"  Missing article files    : {stats['missing_file']}")
    log.info(f"  Output JSONL             : {jsonl_path}")
    log.info(f"  Output CSV               : {csv_path}")
    log.info(f"  Failed log               : {failed_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()