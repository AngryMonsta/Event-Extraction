"""
fde_llm_pipeline
================
End-to-end Fisheries Dispute Event (FDE) extraction pipeline.

Stages
------
  Stage 0  — Text extraction        (text_extraction)
  Stage 1  — Dispute detection      (dispute_detection)
  Stage 2  — FDE data collection    (fde_data_collection)

The LLM is loaded **once** via :mod:`model_registry` and shared across
Stage 1 and Stage 2 so ``from_pretrained`` is called only a single time.

Quick-start
-----------
    from fde_llm_pipeline import (
        # Stage 0
        process_zip, process_folder, process_file,
        process_string,

        # Model (shared across Stage 1 & 2)
        load_shared_model, SharedModel,

        # Stage 1
        DetectionSchema, run_detection, build_system_prompt,

        # Stage 2
        FdeEvent, run_inference, validate_events,

        # Orchestrator
        run_stage0, run_stage1, run_stage2,
    )

CLI entry points
----------------
  python -m fde_llm_pipeline.text_extraction   --help
  python -m fde_llm_pipeline.dispute_detection --help
  python -m fde_llm_pipeline.fde_data_collection --help
  python -m fde_llm_pipeline.run_pipeline      --help
"""

# ---------------------------------------------------------------------------
# Stage 0 — Text Extraction
# ---------------------------------------------------------------------------
from .text_extraction import (
    process_zip,
    process_folder,
    process_file,
    process_docx_file,
    process_doc_file,
    process_txt_file,
    process_string,
    parse_document_for_structured_json,
    extract_body,
)

# ---------------------------------------------------------------------------
# Model Registry  (load-once, shared by Stage 1 & Stage 2)
# ---------------------------------------------------------------------------
from .model_registry import (
    SharedModel,
    load_shared_model,
)

# ---------------------------------------------------------------------------
# Stage 1 — Dispute Detection
# ---------------------------------------------------------------------------
from .dispute_detection import (
    DetectionSchema,
    OutputFileSchema,
    load_model as load_detection_model,   # aliased to avoid collision with Stage 2
    build_system_prompt,
    run_detection,
)

# ---------------------------------------------------------------------------
# Stage 2 — FDE Data Collection
# ---------------------------------------------------------------------------
from .fde_data_collection import (
    FdeCriteria,
    ConflictDrivers,
    FdeEvent,
    SYSTEM_PROMPT as FDE_SYSTEM_PROMPT,   # aliased for clarity
    DRIVER_FIELDS,
    CSV_HEADERS,
    load_model as load_extraction_model,  # aliased to avoid collision with Stage 1
    build_prompt,
    run_inference,
    extract_json_array,
    validate_events,
    event_to_csv_row,
    get_detections,
    is_fde_positive,
)

# ---------------------------------------------------------------------------
# Pipeline Orchestrator (all three stages in one call)
# ---------------------------------------------------------------------------
from .run_pipeline import (
    run_stage0,
    run_stage1,
    run_stage2,
)

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
__all__ = [
    # Stage 0
    "process_zip",
    "process_folder",
    "process_file",
    "process_docx_file",
    "process_doc_file",
    "process_txt_file",
    "process_string",
    "parse_document_for_structured_json",
    "extract_body",
    # Model registry
    "SharedModel",
    "load_shared_model",
    # Stage 1
    "DetectionSchema",
    "OutputFileSchema",
    "load_detection_model",
    "build_system_prompt",
    "run_detection",
    # Stage 2
    "FdeCriteria",
    "ConflictDrivers",
    "FdeEvent",
    "FDE_SYSTEM_PROMPT",
    "DRIVER_FIELDS",
    "CSV_HEADERS",
    "load_extraction_model",
    "build_prompt",
    "run_inference",
    "extract_json_array",
    "validate_events",
    "event_to_csv_row",
    "get_detections",
    "is_fde_positive",
    # Orchestrator
    "run_stage0",
    "run_stage1",
    "run_stage2",
]