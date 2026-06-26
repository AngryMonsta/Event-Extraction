# FDE-LLM-Extraction

This repository contains the built of local pipeline and supporting code used to extract Fisheries Dispute Events (FDEs) from news articles using large language models (LLMs). It includes runnable prototype pipelines, and sample data, experiment notebooks. 

## Prerequisites

- Python 3.10 or newer (3.12 recommended)
- Git
- pip (or an alternative Python package manager)
- Virtual environment tooling (`venv`, `conda`, or similar)
- A CUDA-capable GPU with more than 16 GB of VRAM is strongly recommended for running the local Gemma 4 extraction pipeline
- A Hugging Face account and User Access Token if you plan to download gated models or private checkpoints

## Input Data Acquisition

### Nexis Uni Power Search Query

News articles are sourced from Nexis Uni using the following search parameters:

**Search Query:**
```
Coastal:"dar es salaam" OR "mafia marine park" OR "mafia island" OR (muheza OR tanga OR pangani OR bagamoyo OR mkuranga OR rufiji OR kilwa OR zanzibar OR pemba OR lindi OR mtwara OR tanzania OR tanzanian) AND "exclusive economic zone" OR "territorial waters" OR (coast OR coastline OR coastal OR beach OR island) AND (fish OR fishery OR fisheries OR fisherman OR fishermen OR fisherfolk OR aquaculture OR fishmonger OR mammies OR seine OR cockle)
```

**Query Parameters:**
- **Time Frame:** 1990 - 2017
- **Exclusion:** "dispute"

**Download Instructions:**
Download individual news articles from the Nexis Uni search results and save them to `data`, or use test data in `data/test_data`.

## Environment Setup

1. Clone the repository:

```bash
git clone https://github.gatech.edu/WWF-Team5/FDE-LLM-Extraction.git
cd FDE-LLM-Extraction
```

2. Create and activate a virtual environment:

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Or using conda
conda create -n fde-env python=3.12
conda activate fde-env
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install the pipeline as a package (required to run `python -m fde_llm_pipeline.*` from the repo root):

```bash
pip install -e .
```

5. Authenticate with Hugging Face to access gated models (e.g., Gemma 4):
- Sign in to Hugging Face at https://huggingface.co
- Open your account settings and go to Access Tokens
- Create a new token with the Read role
- Either log in from the command line or export the token as an environment variable

```bash
hf auth login
# or
export HF_TOKEN="hf_..."
```

## Run a quick smoke test using the provided test data:

```bash
python -m fde_llm_pipeline.run_pipeline \
    --zip       data/test_data/test_raw.ZIP \
    --output    results/ \
    --model     google/gemma-4-E4B-it \
    --checkpoint-every 10
```

Run this command from the repository root. The `results/` directory will be created automatically if it does not already exist.

### Running Individual Pipeline Stages

Each stage can also be run independently as a module or via its console-script shortcut:

| Stage | Module | Console script | Purpose |
|-------|--------|----------------|---------|
| 0 – Text extraction | `python -m fde_llm_pipeline.text_extraction` | `fde-extract` | Convert `.docx`/`.doc`/`.txt`/ZIP → JSON |
| 1 – Dispute detection | `python -m fde_llm_pipeline.dispute_detection` | `fde-detect` | Classify articles for FDE/FDA presence |
| 2 – FDE extraction | `python -m fde_llm_pipeline.fde_data_collection` | `fde-collect` | Extract structured FDE records |
| All stages | `python -m fde_llm_pipeline.run_pipeline` | `fde-pipeline` | End-to-end orchestrator |

Append `--help` to any command for full usage details.

## Python API

After `pip install -e .` the package is importable from anywhere:

```python
from fde_llm_pipeline import (
    # Stage 0 — text extraction
    process_zip, process_folder, process_file, process_string,

    # Model registry — load LLM once, share across Stage 1 & 2
    load_shared_model, SharedModel,

    # Stage 1 — dispute detection
    DetectionSchema, run_detection, build_system_prompt,

    # Stage 2 — FDE data collection
    FdeEvent, run_inference, validate_events,

    # Orchestrator
    run_stage0, run_stage1, run_stage2,
)
```

## Folder Structure

```
FDE-LLM-Extraction/
├── pyproject.toml                  ← package metadata & build config (src layout)
├── requirements.txt                ← full pinned dependency list
├── src/
│   └── fde_llm_pipeline/           ← installable Python package
│       ├── __init__.py             ← public API exports
│       ├── text_extraction.py      ← Stage 0: LexisNexis → JSON
│       ├── dispute_detection.py    ← Stage 1: FDE/FDA classification
│       ├── fde_data_collection.py  ← Stage 2: FDE record extraction
│       ├── model_registry.py       ← shared LLM loader (one load for both stages)
│       └── run_pipeline.py         ← end-to-end orchestrator
├── experiments/                    ← experiment scripts and output artifacts
├── data/
│   └── test_data/                  ← prototype inputs for quick iteration
└── results/                        ← CSV / JSONL outputs from pipeline runs
```