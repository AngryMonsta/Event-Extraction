# FDE-LLM-Extraction

This repository contains the local pipeline and supporting code used to extract Fisheries Dispute Events (FDEs) from news articles using large language models (LLMs). It includes runnable prototype pipelines, and sample data, experiment notebooks. The project is organized so you can reproduce experiments, run local extraction jobs, or adapt the pipeline to other document sets.

## Prerequisites

- Python 3.12 or newer
- Git
- pip (or an alternative Python package manager)
- Virtual environment tooling (`venv`, `virtualenv`, or similar)
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
Download individual news articles from the Nexis Uni search results and save them to `data` or use test data in `data/test_data`.

## Environment Setup

1. Clone the repository:

```
git clone https://github.gatech.edu/WWF-Team5/FDE-LLM-Extraction.git
cd FDE-LLM-Extraction
```

2. Create and activate a virtual environment (example using `venv`):

```
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Create a Hugging Face User Access Token and authenticate with it:

   - Sign in to Hugging Face at https://huggingface.co
   - Open your account settings and go to Access Tokens
   - Create a new token with the Read role
   - Either log in from the command line or export the token as an environment variable

```
hf auth login

# or
export HF_TOKEN="hf_..."
```

5. Run a quick smoke test using provided test_data:

```bash
python src/fde_llm_pipeline/run_pipeline.py \
    --zip       data/test_raw.ZIP \
    --output    results/ \
    --model     google/gemma-4-E4B-it \
    --checkpoint-every 10 \
```

Run this command from the repository root. The `results/` directory will be created automatically if it does not already exist.


## Folder Structure

- `experiments/` — Experiment scripts and output artifacts used during development and prototyping.
- `src/` — Core pipeline code and reusable modules for extraction and data handling.
- `data/test_data/` — Local prototype inputs, extracted outputs, and metadata used for quick iteration.
- `results/` — CSV / JSONL result files produced by pipeline or experiments.