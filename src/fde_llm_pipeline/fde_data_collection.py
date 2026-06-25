"""
Stage 2: FDE Extraction Pipeline — Local LLM (Gemma 4, HuggingFace Transformers)
==================================================================================
Reads Stage 1 output summary JSON, filters articles where contains_fde=True,
loads article text from a separate folder, feeds each article to a local
LLM model, validates the output against the FdeEvent Pydantic schema,
and writes both JSONL and CSV outputs.

Directory layout expected:
    stage1_output.json          ← Stage 1 detection results
    articles/
        eDOC_001.json
        eDOC_002.json
        ...

Usage:
    python fde_data_collection.py \\
        --stage1    ../proto_data/DisputeDetect/Gemma4_e4b_summary_test.json \\
        --articles  ../proto_data/TextExtracted/ \\
        --output    ../results/ \\
        --model     google/gemma-4-E4B-it

    # Limit to N articles (for testing)
    python fde_data_collection.py \\
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
import re
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


def _atomic_json_dump(path: Path, payload) -> None:
    """Write JSON atomically so checkpoint files are never partial."""

    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


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
    EcoChngeOther: bool
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
    Location: str = Field(description="Closest identifiable land-based location")
    loc_accuracy: Literal[0, 1, 2, 3, 4, 5] = Field(
        description="0=Country 1=District 2=International 3=NA 4=Region 5=Town"
    )
    Actor1: str = Field(description="Primary actor")
    Actor2: str = Field(description="Secondary actor")
    violence_score: int = Field(
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
            "(violence_score, actors, any true drivers)."
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

--- DEFINITIONS ---
FDE: An incident in which a fisheries resource is contested, disputed, or the
source of conflict between ≥2 ACTORS, at a DISCRETE TEMPORAL MOMENT, in a
DISCRETE LOCATION.
    - Fisheries resource: A natural aquatic resource (e.g., stocks, populations, species, assemblages) which can be
caught by fishing legally. The habitat supporting or containing the resource may also be
considered.
    - Actor: An individual or group engaging in an event. Groups can be organized or spontaneous. All
actors must be people. For example, a hippopotamus attacking fishers for fishing in its territory
is not an FDE because humans are actors on only one side of the conflict.
    - Discrete temporal moment: An event occurs over a defined period of time (days, weeks) versus a systemic condition. For
example, a conflict in June 2018 between two fishers over access to fishing gear would meet
the definition needed to be counted as an FDE, but a description of long-term poverty in a
fishing village is not a discrete temporal moment.
    - Discrete location: An event occurs in a bounded location (landing site, village, place of business) versus being a
diffuse occurrence. For example, conflict at Orion Reef between illegal fishers and marine
authorities would meet the definition needed to be counted as an FDE, but a general description
of dynamite fishing along the coast of T anzania is not a discrete location.

--- REGION CODES ---
0=Arctic
1=Horn of Africa (Tanzania, Kenya, Somalia, Djibouti, Eritrea, Yemen, Ethiopia,
  Sudan, Uganda, Rwanda, Burundi, DRC, Lake Victoria basin, Red Sea,
  Indian Ocean off East Africa)
2=Oceania
3=South America
4=South Asia

--- LOCATION ---
highly specific land-based or coastal points within the event region (e.g., Juani Village, Dar es Salaam, Alula)

--- LOC_ACCURACY CODES ---
0=Country only  1=District/Province  2=International/High-seas
3=NA (indeterminate)  4=Region (named but no district/town)  5=Town/village/named site

--- VIOLENCE SCORE ---
1 = Verbal only — bans, complaints, fines, policy disputes
2 = Physical action, NO bodily harm — property damage, gear/boat confiscation,
    arrests, abductions
3 = Bodily harm — injuries, sexual assault, fatalities
Assign the HIGHEST level met for the event.

--- CONFLICT DRIVERS (true ONLY if explicitly supported by article text) ---
WeakGov        Corruption, weak enforcement, institutional capacity gaps
FishPop        Actual/perceived decline in fish stocks motivating actors
EcoChngeOther   Non-fish ecosystem change (pollution, coral damage, acidification)
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

--- CODING RULES ---
1. Evidence standard: a driver is true if the article states it OR if the
   article contains context that directly implies it (e.g. "fishing unabated
   for five years despite ban" implies WeakGov without using that word).
2. Each distinct event (different date AND location OR actors) = a separate FDE.
3. Aggregate/trend text without a discrete event → no FDE.
4. date: YYYY-MM-DD; use -01 placeholders for unknown month/day.
5. Actor1 = initiating/primary party; Actor2 = responding/secondary party.
6. issue_note: a brief evidence-based summary, for EVERY true driver, quote or paraphrase the article phrase
   that triggered it. This is mandatory — do not assert a driver without
   citing evidence from the text.
7. Return only the JSON array — no extra text before or after.

--- OUTPUT SCHEMA (one object per FDE) ---
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
  "Location": "<string>",
  "loc_accuracy": <0|1|2|3|4|5>,
  "Actor1": "<string>",
  "Actor2": "<string>",
  "violence_score": <1|2|3>,
  "drivers": {
    "WeakGov": <bool>, "FishPop": <bool>, "EcoChngeOther": <bool>, 
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
        batch_size=4,  # Process 4 prompts at once for GPU efficiency
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
    prompts,
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
):
    """Run the model on one or more prompts and return generated text(s).
    
    Args:
        prompts: str or list of str
        
    Returns:
        str if single prompt, list[str] if multiple prompts
    """
    single_prompt = isinstance(prompts, str)
    prompt_list = [prompts] if single_prompt else prompts
    
    outputs = pipe(
        prompt_list,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,   # only the generated part, not the prompt
        pad_token_id=pipe.tokenizer.eos_token_id,
    )

    results: list[str] = []
    # Handle multiple return formats from different transformers versions
    for out in outputs:
        text = None
        # Case: pipeline returns list of dicts per prompt
        if isinstance(out, list):
            # take the first generation entry
            first = out[0] if out else ""
            if isinstance(first, dict):
                text = first.get("generated_text") or first.get("text")
            elif isinstance(first, str):
                text = first
        elif isinstance(out, dict):
            text = out.get("generated_text") or out.get("text")
        elif isinstance(out, str):
            text = out
        else:
            # Fallback to string conversion
            try:
                text = str(out)
            except Exception:
                text = ""

        if text is None:
            text = ""

        results.append(text.strip())

    return results[0] if single_prompt else results


# ---------------------------------------------------------------------------
# JSON extraction & validation
# ---------------------------------------------------------------------------

def _find_matching_bracket(text: str, start_idx: int, open_char: str = '[', close_char: str = ']') -> Optional[int]:
    """Find the index of the matching closing bracket for a bracket at start_idx.
 
    Handles quoted strings and escaped characters. Returns None if not found.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start_idx, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        else:
            if ch == '"':
                in_string = True
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return i
    return None


def _repair_truncated_json(text: str) -> str:
    """Best-effort close open strings and brackets in truncated JSON-like text."""
    stack = []
    in_string = False
    escaped = False
 
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch in ('}', ']') and stack and stack[-1] == ch:
                stack.pop()
 
    repaired = text.rstrip()
    if in_string:
        repaired += '"'
    while stack:
        repaired += stack.pop()
    return repaired

def _sanitise_raw_json(text: str) -> str:
    # 1. Leading-zero integers: match a JSON value position (after : or ,/[)
    #    and replace 0-padded numeric literals with their int value.
    #    Specifically catch the 0000 / 00 etc pattern the model produces.
    text = re.sub(
        r'(?<=[:\s,\[])0+(\d+)',   # 0000 → 0, 007 → 7, 00 → 0
        lambda m: str(int(m.group(0))),
        text,
    )
 
    # 2. Placeholder date strings: "YYYY-MM-DD" variants → "0000-MM-DD"
    #    Handles both digit months ("YYYY-10-15") and literal placeholders ("YYYY-MM-01", "YYYY-MM-DD")
    text = re.sub(
        r'"YYYY-(\d{2}|MM)-(\d{2}|DD)"',
        lambda m: (
            f'"0000-{m.group(1) if m.group(1) != "MM" else "01"}'
            f'-{m.group(2) if m.group(2) != "DD" else "01"}"'
        ),
        text,
    )
    # Catch any remaining fully-symbolic placeholder not matched above
    text = text.replace('"YYYY-MM-DD"', '"0000-01-01"')
 
    return text

def extract_json_array(raw: str) -> Optional[list]:
    """
    Pull a JSON array out of the model's raw output.
    Tries fenced code blocks, direct JSON, balanced-array extraction and
    truncated JSON repair as fallbacks.
    """
    text = raw or ""
 
    # If model uses channel/turn markers, find the channel tag immediately
    # preceding the JSON array or object using a single regex (avoids ambiguous lists).
    # Matches tags like <channel|>, <|channel>, <channel|> etc only when they
    # are directly followed by optional whitespace and either '[' or '{'.
    m_iter = re.finditer(r"<[^>]*channel[^>]*>\s*([\[{])", text, flags=re.IGNORECASE)
    last_m = None
    last_open = None
    for m in m_iter:
        last_m = m
        last_open = m.group(1)
    if last_m is not None:
        start = last_m.end() - 1  # position of '[' or '{'
        open_char = last_open
        close_char = ']' if open_char == '[' else '}'
        end = _find_matching_bracket(text, start, open_char=open_char, close_char=close_char)
        if end is not None:
            candidate = text[start:end+1]
            try:
                return json.loads(_sanitise_raw_json(candidate))
            except json.JSONDecodeError:
                repaired = _repair_truncated_json(_sanitise_raw_json(candidate))
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
        else:
            # truncated: take remainder starting at the opening char and attempt repair
            candidate = text[start:]
            repaired = _repair_truncated_json(_sanitise_raw_json(candidate))
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
 
    # 1. Try fenced code blocks
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                return json.loads(_sanitise_raw_json(cleaned))
            except json.JSONDecodeError:
                repaired = _repair_truncated_json(_sanitise_raw_json(cleaned))
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    continue
 
    # 2. Try raw JSON
    try:
        return json.loads(_sanitise_raw_json(text))
    except json.JSONDecodeError:
        pass
 
    # 3. Find balanced JSON array starting at first '[' (fallback)
    start = text.find('[')
    if start != -1:
        end = _find_matching_bracket(text, start, open_char='[', close_char=']')
        if end is not None:
            candidate = text[start:end+1]
            try:
                return json.loads(_sanitise_raw_json(candidate))
            except json.JSONDecodeError:
                repaired = _repair_truncated_json(_sanitise_raw_json(candidate))
                try:
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    pass
 
    # 4. As a last resort, strip common channel/turn/eos tokens and retry
    cleaned = re.sub(r"<\/?\|?\w+\|?>", "", text)
    cleaned = re.sub(r"<(?:eos|turn|\|eos\||\|turn\|)>", "", cleaned)
    try:
        return json.loads(_sanitise_raw_json(cleaned))
    except json.JSONDecodeError:
        repaired = _repair_truncated_json(_sanitise_raw_json(cleaned))
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
 
    # 5. Try single JSON object
    obj_start = text.find('{')
    if obj_start != -1:
        obj_end = _find_matching_bracket(text, obj_start, open_char='{', close_char='}')
        if obj_end is not None:
            candidate = text[obj_start:obj_end+1]
            try:
                parsed = json.loads(_sanitise_raw_json(candidate))
                return [parsed]
            except json.JSONDecodeError:
                repaired = _repair_truncated_json(_sanitise_raw_json(candidate))
                try:
                    parsed = json.loads(repaired)
                    return [parsed]
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

        # Normalize common omissions and types before validation
        ev = _normalize_event(ev, article_id)
        try:
            fde = FdeEvent(**ev)
            valid.append(fde.model_dump())
        except ValidationError as e:
            ev["_validation_error"] = str(e)
            ev["_source_article_id"] = article_id
            invalid.append(ev)
    return valid, invalid


def _normalize_event(ev: dict, article_id: str) -> dict:
    """Fill missing driver/criteria keys and coerce simple types to help validation.

    This attempts non-destructive repairs so Pydantic validation can succeed
    for common model omissions produced by the LLM.
    """
    # Ensure drivers dict has all expected keys
    drv = ev.get("drivers") or {}
    for k in DRIVER_FIELDS:
        if k not in drv:
            drv[k] = False
    ev["drivers"] = drv

    # Ensure fde_criteria present with boolean fields
    crit = ev.get("fde_criteria") or {}
    for key in ("min_two_actors", "discrete_temporal_moment", "discrete_location", "fisheries_resource_contested"):
        crit.setdefault(key, False)
    ev["fde_criteria"] = crit

    # Coerce numeric-like fields
    if "year" in ev and ev["year"] not in (None, ""):
        try:
            ev["year"] = int(ev["year"])
        except Exception:
            ev.pop("year", None)

    if "violence_score" in ev and ev["violence_score"] not in (None, ""):
        try:
            ev["violence_score"] = int(ev["violence_score"])
        except Exception:
            ev.pop("violence_score", None)

    for key in ("region", "loc_accuracy"):
        if key in ev and ev[key] not in (None, ""):
            try:
                ev[key] = int(ev[key])
            except Exception:
                ev.pop(key, None)

    # Infer year from date when missing / zero
    date = ev.get("date", "")
    if isinstance(date, str):
        # Normalise any remaining YYYY placeholder to 0000
        if date.startswith("YYYY"):
            date = "0000" + date[4:]
            ev["date"] = date
 
        year_val = ev.get("year")
        if not year_val:                      
            if len(date) >= 4 and date[:4].isdigit():
                try:
                    ev["year"] = int(date[:4])
                except Exception:
                    pass

    # Ensure article_id present
    if "article_id" not in ev or not ev["article_id"]:
        ev["article_id"] = article_id

    return ev


# ---------------------------------------------------------------------------
# Flat CSV row builder
# ---------------------------------------------------------------------------

# All driver field names in order
DRIVER_FIELDS = [
    "WeakGov", "FishPop", "EcoChngeOther", "GroundsLim", "ForeignFisher",
    "IllegalFishing", "IncrEfficiency", "OpsScales", "Markets",
    "FoodInsecurity", "Marginalization", "StratLoc", "MaritimeCrime"
]

CSV_HEADERS = [
    "article_id", "date", "year", "country", "region",
    "Location", "loc_accuracy", "Actor1", "Actor2", "violence_score",
    # "fde_min_two_actors", "fde_discrete_temporal_moment",
    # "fde_discrete_location", "fde_fisheries_resource_contested",
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
        "Location":      ev.get("Location", ""),
        "loc_accuracy":  ev.get("loc_accuracy", ""),
        "Actor1":        ev.get("Actor1", ""),
        "Actor2":        ev.get("Actor2", ""),
        "violence_score": ev.get("violence_score", ""),
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
                        help="Path to folder containing article .json files")
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
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Write a checkpoint after this many processed articles (default: 50)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for GPU inference (default: 8)")
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
    checkpoint_path = output_dir / f"fde_checkpoint_{run_ts}.json"

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

    all_results: list[dict] = []
    pending_batch: list[dict] = []

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

    # ── Batch inference preparation ────────────────────────────────────────
    # Accumulate articles, prompts, and metadata for batched inference
    batch_articles = []
    batch_prompts = []
    batch_metadata = []

    def _process_batch():
        """Run inference on accumulated batch and write results."""
        if not batch_articles:
            return
        
        # Run batched inference
        raw_outputs = run_inference(
            pipe, batch_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        
        # Process each result
        for metadata, raw in zip(batch_metadata, raw_outputs):
            article_id = metadata["article_id"]
            elapsed = metadata.get("elapsed", 0)
            log.info(f"  Inference: {elapsed:.1f}s  |  Raw output length: {len(raw)} chars")
            
            parsed = extract_json_array(raw)
            
            if parsed is None:
                log.warning(f"  JSON parse failed — logging to {failed_path.name}")
                failed_fh.write(json.dumps({
                    "article_id": article_id,
                    "error":      "json_parse_failed",
                    "raw_output": raw,
                }) + "\n")
                failed_fh.flush()
                stats["parse_failed"] += 1
                stats["processed"] += 1
                all_results.append({
                    "article_id": article_id,
                    "status": "json_parse_failed",
                    "raw_output": raw,
                })
                pending_batch.append(all_results[-1])
                continue
            
            if not isinstance(parsed, list) or len(parsed) == 0:
                log.info(f"  Model returned empty array (no FDEs found in article)")
                stats["no_events"] += 1
                stats["processed"] += 1
                all_results.append({
                    "article_id": article_id,
                    "status": "no_events",
                    "events": [],
                })
                pending_batch.append(all_results[-1])
                continue
            
            valid_events, invalid_events = validate_events(parsed, article_id)
            
            for ev in invalid_events:
                log.warning(f"  Validation error on 1 event — logged to failed file")
                failed_fh.write(json.dumps({
                    "article_id": article_id,
                    "error":      "validation_error",
                    "raw_event":  ev,
                }) + "\n")
                failed_fh.flush()
                stats["val_errors"] += 1
            
            for ev in valid_events:
                jsonl_fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                csv_writer.writerow(event_to_csv_row(ev))
            
            jsonl_fh.flush()
            csv_fh.flush()
            
            stats["events_found"] += len(valid_events)
            stats["processed"] += 1
            
            all_results.append({
                "article_id": article_id,
                "status": "ok",
                "valid_events": valid_events,
                "invalid_events": invalid_events,
            })
            pending_batch.append(all_results[-1])
            
            log.info(
                f"  Valid events: {len(valid_events)}  |  "
                f"Invalid: {len(invalid_events)}  |  "
                f"Running total: {stats['events_found']}"
            )
        
        # Checkpoint if batch threshold reached
        if len(pending_batch) >= args.checkpoint_every:
            _atomic_json_dump(checkpoint_path, all_results)
            log.info(
                f"  Checkpoint saved -> {checkpoint_path.name} "
                f"({len(all_results)} articles, batch={len(pending_batch)})"
            )
            pending_batch.clear()
        
        batch_articles.clear()
        batch_prompts.clear()
        batch_metadata.clear()

    # ── Main loop ──────────────────────────────────────────────────────────
    for idx, entry in enumerate(fde_articles, 1):
        article_id = entry["article_id"]
        text_file   = articles_dir / f"{article_id}.json"

        log.info(f"[{idx}/{stats['total']}] {article_id}")

        # ── Load article text ──────────────────────────────────────────────
        if not text_file.exists():
            log.warning(f"  Article file not found: {text_file} — skipping")
            failed_fh.write(json.dumps({
                "article_id": article_id,
                "error": "file_not_found",
                "path": str(text_file),
            }) + "\n")
            failed_fh.flush()
            stats["missing_file"] += 1
            continue

        article_text = text_file.read_text(encoding="utf-8", errors="replace").strip()

        # ── Accumulate for batch inference ─────────────────────────────────
        t0 = time.time()
        prompt = build_prompt(pipe.tokenizer, article_id, article_text)
        batch_articles.append(article_id)
        batch_prompts.append(prompt)
        batch_metadata.append({
            "article_id": article_id,
            "text": article_text,
            "elapsed": time.time() - t0,
        })

        # ── Process batch if we've accumulated enough articles ──────────────
        if len(batch_articles) >= args.batch_size or idx == stats["total"]:
            _process_batch()

    # ── Close files ────────────────────────────────────────────────────────
    jsonl_fh.close()
    failed_fh.close()
    csv_fh.close()

    if all_results:
        _atomic_json_dump(checkpoint_path, all_results)

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
    log.info(f"  Checkpoint               : {checkpoint_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()