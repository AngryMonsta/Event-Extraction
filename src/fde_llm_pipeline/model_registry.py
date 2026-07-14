"""
model_registry.py
-----------------
Loads the LLM **once** and returns the shared components that both
Stage 1 and Stage 2 need.

Stage 1 (dispute_detection) drives the model through AutoProcessor +
raw model.generate() calls — it needs (processor, model).

Stage 2 (fde_extraction) drives it through a transformers
text-generation pipeline — it needs a pipeline object whose .model and
.tokenizer attributes point at the *same already-loaded* objects.

This module owns the single model load and builds both views of it so
neither downstream stage has to touch torch.load or HuggingFace's
from_pretrained.

Public API
----------
    from model_registry import load_shared_model, SharedModel

    shared = load_shared_model(
        model_name   = "google/gemma-4-E4B-it",
        device       = "auto",           # "auto" | "cpu" | "cuda:0"
        device_map_mode = "single",      # "single" | "auto" (multi-GPU)
    )

    # Stage 1 consumers
    shared.processor   # AutoProcessor
    shared.model       # AutoModelForCausalLM (.eval(), bfloat16)

    # Stage 2 consumers
    shared.pipe        # transformers text-generation Pipeline
                       # (same model & tokenizer objects, no second load)
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class SharedModel:
    """
    Container for the single loaded model and its two interface views.

    Attributes
    ----------
    model_name : str
        The HuggingFace model ID that was loaded.
    model : AutoModelForCausalLM
        The underlying causal LM (eval mode, bfloat16 on CUDA).
    processor : AutoProcessor
        Used by Stage 1 for chat-template formatting and tokenisation.
    tokenizer : AutoTokenizer
        The tokenizer component (also accessible via processor.tokenizer,
        kept here as a convenience reference for Stage 2).
    pipe : transformers Pipeline
        A text-generation pipeline wrapping the *same* model and tokenizer
        objects.  No second load occurs.
    device : str
        Resolved device string (e.g. "cuda:0" or "cpu").
    """

    model_name: str
    model: Any          # AutoModelForCausalLM
    processor: Any      # AutoProcessor
    tokenizer: Any      # AutoTokenizer (same object as processor.tokenizer)
    pipe: Any           # transformers Pipeline
    device: str


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_shared_model(
    model_name: str = "google/gemma-4-E4B-it",
    device: str = "auto",
    device_map_mode: str = "single",
    stage2_batch_size: int = 4,
) -> SharedModel:
    """
    Load the LLM and build both Stage-1 and Stage-2 interface views.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. 
        Use "google/gemma-4-E4B-it-qat-mobile-transformers" if you don't have enough GPU memory.
    device : str
        "auto" lets torch choose the best visible device.
        Pass "cpu", "cuda", or "cuda:N" to pin explicitly.
    device_map_mode : str
        "single"  — put the whole model on one device (default).
        "auto"    — shard across multiple GPUs via Accelerate device_map.
    stage2_batch_size : int
        Batch size baked into the Stage 2 text-generation pipeline.
    """

    t0 = time.time()

    #  Resolve device & dtype
    if device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda:0"
            dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
            dtype = torch.float16
        else:
            resolved_device = "cpu"
            dtype = torch.float32

    else:
        resolved_device = device
    
    is_mps = (
        resolved_device == "mps"
        or (isinstance(resolved_device, torch.device) and resolved_device.type == "mps")
    )

    if is_mps:
        device_map = None
    elif device_map_mode == "auto":
        device_map = "auto"
    else:
        device_map = {"": resolved_device}

    log.info("=" * 60)
    log.info(f"[ModelRegistry] Loading model : {model_name}")
    log.info(f"[ModelRegistry] dtype={dtype}  device_map={device_map}  resolved_device={resolved_device}")
    log.info("=" * 60)

    #  Load processor (Stage 1 primary interface)
    log.info("[ModelRegistry] Loading processor …")
    processor = AutoProcessor.from_pretrained(model_name)
    log.info(f"[ModelRegistry] Processor ready  ({time.time() - t0:.1f}s)")

    #  Load model weights (load once, shared by both stages)
    log.info("[ModelRegistry] Loading model weights …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    if is_mps:
        model = model.to(resolved_device)
    
    model.eval()
    log.info(
        f"[ModelRegistry] Model ready  ({time.time() - t0:.1f}s)  "
        f"device={next(model.parameters()).device}"
    )

    #  Resolve tokenizer reference 
    if hasattr(processor, "tokenizer"):
        tokenizer = processor.tokenizer
    else:
        log.info(
            "[ModelRegistry] processor has no .tokenizer attr — "
            "loading tokenizer separately (fast, no weight load)"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    #  Build Stage 2 pipeline
    log.info("[ModelRegistry] Building Stage-2 text-generation pipeline …")
    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if device_map_mode == "auto" else None,
        batch_size=stage2_batch_size,
    )
    log.info(f"[ModelRegistry] Pipeline ready  ({time.time() - t0:.1f}s)")

    log.info(f"[ModelRegistry] Total load time : {time.time() - t0:.1f}s")

    return SharedModel(
        model_name=model_name,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        pipe=pipe,
        device=resolved_device,
    )
