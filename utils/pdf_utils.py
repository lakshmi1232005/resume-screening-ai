"""
utils/pdf_utils.py
====================
Utilities for extracting raw text from PDF files.

Why two PDF libraries?
    - pdfplumber: excellent at preserving layout/whitespace, great for
      most "normal" text-based resumes.
    - PyMuPDF (imported as `fitz`): faster and sometimes succeeds on PDFs
      that trip up pdfplumber (unusual encodings, certain generators).

    We try pdfplumber FIRST. If it returns little/no text, we fall back
    to PyMuPDF. If BOTH return almost nothing, the PDF is very likely a
    "scanned" image-based PDF (i.e. a photo/scan of a resume with no
    real text layer) -- which neither library can read, because there
    is no text to extract, only pixels. We raise a clear error in that
    case rather than silently returning an empty resume.
"""

import fitz  # PyMuPDF
import pdfplumber

# If extracted text has fewer than this many non-whitespace characters,
# we treat the extraction as "failed" and try the next strategy.
MIN_VALID_TEXT_LENGTH = 30


class PDFExtractionError(Exception):
    """Raised when we cannot extract usable text from a PDF."""
    pass


def _extract_with_pdfplumber(file_path: str) -> str:
    """Attempt text extraction using pdfplumber. Returns '' on failure."""
    text_chunks = []
    try:
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                return ""  # Empty PDF (0 pages)
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)
    except Exception:
        # pdfplumber can raise various low-level parsing errors on
        # malformed PDFs -- we swallow them here and let the caller
        # fall back to PyMuPDF instead of crashing the whole app.
        return ""
    return "\n".join(text_chunks).strip()


def _extract_with_pymupdf(file_path: str) -> str:
    """Attempt text extraction using PyMuPDF. Returns '' on failure."""
    text_chunks = []
    try:
        doc = fitz.open(file_path)
        if doc.page_count == 0:
            return ""
        for page in doc:
            text_chunks.append(page.get_text())
        doc.close()
    except Exception:
        return ""
    return "\n".join(text_chunks).strip()


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract raw text from a PDF file, trying pdfplumber first and
    falling back to PyMuPDF if needed.

    Args:
        file_path: Path to the PDF file on disk.

    Returns:
        str: The extracted raw text (not yet cleaned/preprocessed).

    Raises:
        PDFExtractionError: If no usable text could be extracted at all
                             (e.g. the PDF is empty, corrupted, or a
                             scanned image with no text layer).
    """
    # --- Attempt 1: pdfplumber (preferred -- best layout fidelity) ---
    text = _extract_with_pdfplumber(file_path)

    # --- Attempt 2: PyMuPDF fallback, only if pdfplumber came up short ---
    if len(text.strip()) < MIN_VALID_TEXT_LENGTH:
        fallback_text = _extract_with_pymupdf(file_path)
        if len(fallback_text.strip()) > len(text.strip()):
            text = fallback_text

    # --- Final check: did we get anything usable at all? ---
    if len(text.strip()) < MIN_VALID_TEXT_LENGTH:
        raise PDFExtractionError(
            "Could not extract readable text from this PDF. It may be:\n"
            "  - An empty PDF (0 pages or blank pages)\n"
            "  - A scanned/image-based PDF with no real text layer "
            "(needs OCR, which this tool does not currently perform)\n"
            "  - A corrupted or password-protected file\n"
            "Please upload a text-based PDF resume instead."
        )

    return text


def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from a plain .txt file (used for job descriptions).

    Args:
        file_path: Path to the .txt file.

    Returns:
        str: The file's text content.

    Raises:
        PDFExtractionError: If the file is empty or unreadable.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
    except Exception as exc:
        raise PDFExtractionError(f"Could not read text file: {exc}") from exc

    if len(text) < MIN_VALID_TEXT_LENGTH:
        raise PDFExtractionError(
            "The uploaded text file appears to be empty or too short to be "
            "a valid job description."
        )
    return text
