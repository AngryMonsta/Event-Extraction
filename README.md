# FDE-LLM-Extraction

This repository contains the local pipeline and supporting code used to extract Fisheries Dispute Events (FDEs) from news articles using large language models (LLMs). It includes runnable prototype pipelines, and sample data, experiment notebooks. The project is organized so you can reproduce experiments, run local extraction jobs, or adapt the pipeline to other document sets.

## Prerequisites

- Python 3.12 or newer
- Git
- pip (or an alternative Python package manager)
- Virtual environment tooling (`venv`, `virtualenv`, or similar)
- A CUDA-capable GPU with more than 16 GB of VRAM is strongly recommended for running the local Gemma 4 extraction pipeline
- A Hugging Face account and User Access Token if you plan to download gated models or private checkpoints

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

   - Sign in to Hugging Face at https://huggingface.com
   - Open your account settings and go to Access Tokens
   - Create a new token with the Read role
   - Either log in from the command line or export the token as an environment variable

```
hf auth login

# or
export HF_TOKEN="hf_..."
```

5. Run a quick smoke test or prototype script (examples are under `experiments/` and `src/`):

```bash
python src/fde_local_pipeline.py \
    --stage1 src/proto_data/Gemma4_e4b_summary_2.json \
    --articles src/proto_data/n_extracted/ \
    --output results/ \
    --model google/gemma-4-E4B-it
```

Run this command from the repository root. The `results/` directory will be created automatically if it does not already exist.


## Folder Structure

- `experiments/` — Experiment scripts and output artifacts used during development and prototyping.
- `src/` — Core pipeline code and reusable modules for extraction and data handling.
    - `proto_data/` — Local prototype inputs, extracted outputs, and metadata used for quick iteration.
- `results/` — CSV / JSONL result files produced by experiments.