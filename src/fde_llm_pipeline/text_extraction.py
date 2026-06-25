"""
text_extraction.py
----------------------
Extracts and structures article text from LexisNexis news exports.

Supported input modes
---------------------
  Mode            | Function              | CLI flag
  ----------------|-----------------------|---------------------------
  ZIP archive     | process_zip()         | --zip  path/to/file.ZIP
  Folder          | process_folder()      | --folder path/to/dir/
  Single .docx    | process_docx_file()   | --file path/to/file.docx
  Single .doc     | process_doc_file()    | --file path/to/file.doc
  Single .txt     | process_txt_file()    | --file path/to/file.txt
  Raw string      | process_string()      | (Python API only)

Mixed folders (containing any combination of .docx / .doc / .txt) are
handled automatically by process_folder().

Each extracted article is saved as a JSON file:
    e<DOC_ID>.json  →  {doc_id, title, source, load_date, article_text}

The `article_text` field contains the body text only (stripped of the
LexisNexis header/footer).  When the Body…Load-Date pattern is not
present (e.g. plain .txt files or raw strings), the full text is kept.

Downstream usage — import
-------------------------
    from text_extraction import (
        process_zip, process_folder,
        process_docx_file, process_doc_file,
        process_txt_file, process_string,
    )

    # Single string (no file I/O required)
    record = process_string("Article title\\nReuters\\nJanuary 1, 2005\\nBody\\n...", doc_id="DOC_001")

    # Write to disk
    process_docx_file("article.docx", output_dir="data/stage0", doc_id="DOC_042")

Downstream usage — CLI
----------------------
    python text_extraction.py --zip    Somalia500.ZIP  --out data/stage0          # IDs start at DOC_0001
    python text_extraction.py --zip    Somalia500.ZIP  --out data/stage0 --id-start 501  # IDs start at DOC_0501
    python text_extraction.py --folder Files1-500        --out data/stage0
    python text_extraction.py --file   article.docx      --out data/stage0
    python text_extraction.py --file   article.doc       --out data/stage0
    python text_extraction.py --file   article.txt       --out data/stage0

Dependencies
------------
    pip install python-docx tqdm
    # LibreOffice must be installed for .doc support (used via subprocess)
"""

import os
import re
import json
import datetime
import argparse
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
 
from docx import Document
from docx.document import Document as DocumentType
from tqdm import tqdm
 
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
DOCLIST_PATTERN = re.compile(r"doclist", re.IGNORECASE)
LOAD_DATE_RE    = re.compile(r"Load-Date:\s*([A-Za-z]+ \d{1,2}, \d{4})")
BODY_RE         = re.compile(r"Body\s*(.*?)\s*Load-Date:", re.DOTALL | re.IGNORECASE)
 
SUPPORTED_EXTENSIONS = {".docx", ".doc", ".txt"}
 
 
# ---------------------------------------------------------------------------
# Low-level text extractors  (one per format)
# ---------------------------------------------------------------------------
 
def _text_from_docx(source) -> str:
    """
    Return plain text from a .docx file path or an open Document object.
 
    Parameters
    ----------
    source : str | Path | Document
        File path or already-opened python-docx Document.
    """
    try:
        doc = source if isinstance(source, DocumentType) else Document(str(source))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as exc:
        return f"Error extracting text: {exc}"
 
 
def _text_from_doc(path: str | Path) -> str:
    """
    Return plain text from a legacy .doc file by converting it to .docx
    via LibreOffice, then reading the result.
 
    LibreOffice must be installed and on PATH.  Conversion is done in a
    temporary directory so the original file is never modified.
 
    Parameters
    ----------
    path : str | Path
        Path to the .doc file.
    """
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmp:
        # LibreOffice writes the converted file into `tmp`
        result = subprocess.run(
            [
                "libreoffice", "--headless", "--convert-to", "docx",
                "--outdir", tmp, str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return f"Error converting .doc to .docx: {result.stderr.strip()}"
 
        converted = Path(tmp) / (path.stem + ".docx")
        if not converted.exists():
            return f"Error: LibreOffice did not produce {converted}"
 
        return _text_from_docx(converted)
 
 
def _text_from_txt(path: str | Path, encoding: str = "utf-8") -> str:
    """
    Return the raw text content of a .txt file.
 
    Falls back to latin-1 if utf-8 decoding fails.
 
    Parameters
    ----------
    path : str | Path
    encoding : str
        Primary encoding to try (default: utf-8).
    """
    path = Path(path)
    try:
        return path.read_text(encoding=encoding)
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
    except Exception as exc:
        return f"Error reading text file: {exc}"
 
 
# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------
 
def extract_body(text: str) -> str | None:
    """
    Pull the article body from a LexisNexis-formatted string.
 
    LexisNexis exports delimit the body with 'Body' … 'Load-Date:'.
    Returns None if the pattern is absent (e.g. plain .txt inputs).
    """
    match = BODY_RE.search(text)
    return match.group(1).strip() if match else None
 
 
def _parse_date(date_str: str) -> datetime.date | None:
    """Try common LexisNexis date formats; return a date object or None."""
    for fmt in ("%B %d, %Y %A", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None
 
 
def parse_document_for_structured_json(doc_id: str, article_text: str) -> dict:
    """
    Parse a raw article string into a structured record.
 
    Heuristic header layout (LexisNexis .docx / .doc exports):
      line 0 → article title
      line 1 → publication source
      line 2 → publication date  (e.g. "January 15, 2003 Wednesday")
      Load-Date: <date>  — fallback when line 2 cannot be parsed
 
    For plain .txt files or raw strings that don't follow this layout,
    title/source/load_date default to sentinel values and article_text
    contains the full text (no Body stripping attempted).
 
    Parameters
    ----------
    doc_id : str
        Unique document identifier, e.g. "DOC_0042".
    article_text : str
        Raw extracted text.
 
    Returns
    -------
    dict
        Keys: doc_id, title, source, load_date (ISO-8601 or "Unknown Date"),
        article_text (body only, or full text if no Body marker found).
    """
    title     = "Untitled Document"
    source    = "Unknown Source"
    load_date = "Unknown Date"
 
    lines = [ln.strip() for ln in article_text.split("\n") if ln.strip()]
 
    if lines:
        title  = lines[0]
        source = lines[1] if len(lines) > 1 else source
 
        # Try line 2 as a publication date
        parsed_date = _parse_date(lines[2]) if len(lines) > 2 else None
 
        # Fall back to the explicit Load-Date field
        if parsed_date is None:
            ld_match = LOAD_DATE_RE.search(article_text)
            if ld_match:
                parsed_date = _parse_date(ld_match.group(1))
 
        if parsed_date is not None:
            load_date = parsed_date.isoformat()
 
    # Body extraction: use LexisNexis delimiters when present, else full text
    body = extract_body(article_text)
    if body is None:
        body = article_text.strip() or None
 
    return {
        "doc_id":       doc_id,
        "title":        title,
        "source":       source,
        "load_date":    load_date,
        "article_text": body,
    }
 
 
# ---------------------------------------------------------------------------
# Output helper
# ---------------------------------------------------------------------------
 
def _write_record(record: dict, output_dir: str) -> None:
    """Write a record dict to <output_dir>/e<doc_id>.json."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"e{record['doc_id']}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=4)
 
 
# ---------------------------------------------------------------------------
# Single-item processors  (importable building blocks)
# ---------------------------------------------------------------------------
 
def process_string(
    text: str,
    doc_id: str = "DOC_0001",
    output_dir: str | None = None,
) -> dict:
    """
    Parse a raw article string and optionally write a JSON record.
 
    Parameters
    ----------
    text : str
        Raw article text (LexisNexis-formatted or plain).
    doc_id : str
        Identifier to assign this record.
    output_dir : str | None
        If given, write the JSON record to this directory.
 
    Returns
    -------
    dict
        Structured record (also written to disk when output_dir is set).
    """
    record = parse_document_for_structured_json(doc_id, text)
    if output_dir:
        _write_record(record, output_dir)
    return record
 
 
def process_docx_file(
    path: str | Path,
    doc_id: str = "DOC_0001",
    output_dir: str | None = None,
) -> dict:
    """
    Extract and structure a single .docx file.
 
    Parameters
    ----------
    path : str | Path
    doc_id : str
    output_dir : str | None
        Write JSON to this directory when provided.
 
    Returns
    -------
    dict
    """
    text   = _text_from_docx(path)
    record = parse_document_for_structured_json(doc_id, text)
    if output_dir:
        _write_record(record, output_dir)
    return record


def process_doc_file(
    path: str | Path,
    doc_id: str = "DOC_0001",
    output_dir: str | None = None,
) -> dict:
    """
    Extract and structure a single legacy .doc file (requires LibreOffice).

    Parameters
    ----------
    path : str | Path
    doc_id : str
    output_dir : str | None
        Write JSON to this directory when provided.

    Returns
    -------
    dict
    """
    text   = _text_from_doc(path)
    record = parse_document_for_structured_json(doc_id, text)
    if output_dir:
        _write_record(record, output_dir)
    return record


def process_txt_file(
    path: str | Path,
    doc_id: str = "DOC_0001",
    output_dir: str | None = None,
    encoding: str = "utf-8",
) -> dict:
    """
    Extract and structure a single plain-text file.

    Parameters
    ----------
    path : str | Path
    doc_id : str
    output_dir : str | None
        Write JSON to this directory when provided.
    encoding : str
        Primary encoding; falls back to latin-1 on failure.

    Returns
    -------
    dict
    """
    text   = _text_from_txt(path, encoding=encoding)
    record = parse_document_for_structured_json(doc_id, text)
    if output_dir:
        _write_record(record, output_dir)
    return record


# ---------------------------------------------------------------------------
# Batch processors
# ---------------------------------------------------------------------------

def process_zip(
    zip_path: str,
    output_dir: str,
    id_start: int = 1,
) -> int:
    """
    Extract and structure all supported articles from a ZIP archive.

    Document IDs are assigned sequentially starting from `id_start`,
    based on the count of eligible files inside the ZIP (skipping
    non-article entries such as doclist manifests).

    Parameters
    ----------
    zip_path : str
    output_dir : str
    id_start : int
        Starting integer for DOC_XXXX IDs (default: 1).

    Returns
    -------
    int  Number of articles written.
    """
    os.makedirs(output_dir, exist_ok=True)

    with ZipFile(zip_path) as zf:
        names = zf.namelist()
        total = len(names)
        print(f"ZIP contains {total} entries in total")

    count = 0
    with ZipFile(zip_path) as zf:
        names = zf.namelist()
        for filename in tqdm(names, total=total, desc="ZIP → JSON"):
            ext = Path(filename).suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue
            if DOCLIST_PATTERN.search(filename):
                continue

            unique_id = f"DOC_{(id_start + count):04d}"

            if ext == ".docx":
                with zf.open(filename) as fh:
                    doc  = Document(BytesIO(fh.read()))
                text = _text_from_docx(doc)
            elif ext == ".doc":
                # Extract to a temp file, convert, then clean up
                with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
                    tmp.write(zf.read(filename))
                    tmp_path = tmp.name
                try:
                    text = _text_from_doc(tmp_path)
                finally:
                    os.unlink(tmp_path)
            else:  # .txt
                text = zf.read(filename).decode("utf-8", errors="replace")

            record = parse_document_for_structured_json(unique_id, text)
            _write_record(record, output_dir)
            count += 1

    print(f"Extracted {count} articles from ZIP → {output_dir}")
    return count


def process_folder(
    folder_path: str,
    output_dir: str,
    id_start: int = 1,
) -> int:
    """
    Extract and structure all supported files in a flat folder.

    Accepts any mix of .docx, .doc, and .txt files.  Files whose names
    contain 'doclist' are skipped (LexisNexis manifest files).

    Parameters
    ----------
    folder_path : str
    output_dir : str
    id_start : int
        Starting integer for DOC_XXXX IDs.

    Returns
    -------
    int  Number of articles written.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(
        f for f in os.listdir(folder_path)
        if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        and not DOCLIST_PATTERN.search(f)
    )

    if not all_files:
        print(f"No supported files (.docx / .doc / .txt) found in {folder_path}")
        return 0

    print(f"Found {len(all_files)} file(s) in {folder_path}")

    count = 0
    for i, filename in tqdm(enumerate(all_files), total=len(all_files), desc="Folder → JSON"):
        unique_id = f"DOC_{(id_start + i):04d}"
        full_path = os.path.join(folder_path, filename)
        ext       = Path(filename).suffix.lower()

        if ext == ".docx":
            text = _text_from_docx(full_path)
        elif ext == ".doc":
            text = _text_from_doc(full_path)
        else:
            text = _text_from_txt(full_path)

        record = parse_document_for_structured_json(unique_id, text)
        _write_record(record, output_dir)
        count += 1

    print(f"Extracted {count} articles → {output_dir}")
    return count


def process_file(
    path: str | Path,
    output_dir: str,
    doc_id: str | None = None,
) -> dict:
    """
    Detect file type and extract a single file of any supported format.

    This is a convenience dispatcher — it calls process_docx_file,
    process_doc_file, or process_txt_file based on the file extension.

    Parameters
    ----------
    path : str | Path
    output_dir : str
    doc_id : str | None
        If None, derived from the filename stem.

    Returns
    -------
    dict  The structured record.
    """
    path = Path(path)
    ext  = path.suffix.lower()
    if doc_id is None:
        doc_id = f"DOC_{path.stem}"

    if ext == ".docx":
        return process_docx_file(path, doc_id=doc_id, output_dir=output_dir)
    elif ext == ".doc":
        return process_doc_file(path, doc_id=doc_id, output_dir=output_dir)
    elif ext == ".txt":
        return process_txt_file(path, doc_id=doc_id, output_dir=output_dir)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract LexisNexis news articles to structured JSON.\n\n"
            "Supported input formats: .docx, .doc (requires LibreOffice), .txt\n"
            "Supported input modes:   --zip, --folder, --file"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--zip",    metavar="PATH", help="ZIP archive containing article files")
    group.add_argument("--folder", metavar="PATH", help="Directory of article files (.docx/.doc/.txt)")
    group.add_argument("--file",   metavar="PATH", help="Single article file (.docx, .doc, or .txt)")

    parser.add_argument(
        "--out", metavar="DIR", required=True,
        help="Output directory for JSON files"
    )
    parser.add_argument(
        "--doc-id", metavar="ID", default=None,
        help="(--file mode) Explicit DOC ID; defaults to the filename stem"
    )
    parser.add_argument(
        "--id-start", type=int, default=1,
        help="Starting integer for DOC_XXXX IDs (default: 1); applies to --zip and --folder modes"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.zip:
        process_zip(args.zip, args.out, id_start=args.id_start)
    elif args.folder:
        process_folder(args.folder, args.out, id_start=args.id_start)
    else:
        process_file(args.file, output_dir=args.out, doc_id=args.doc_id)


if __name__ == "__main__":
    main()
