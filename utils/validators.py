"""
utils/validators.py
=====================
Validation helpers for uploaded files.

Why this file exists:
    Before we try to parse ANY uploaded file, we should check that it's
    a reasonable file to begin with -- right extension, not too large,
    actually exists on disk. Doing these checks up front means our
    parsing code (Phase 2+) doesn't have to worry about garbage input,
    and the user gets a clear, friendly error message instead of a
    confusing crash.
"""

import os

from config import MAX_FILE_SIZE_MB


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""
    pass


def validate_file(file_path: str, allowed_extensions: tuple) -> None:
    """
    Validate that a file exists, has an allowed extension, and isn't
    larger than the configured size limit.

    Args:
        file_path: Path to the file on disk.
        allowed_extensions: Tuple of allowed extensions, e.g. (".pdf",).

    Raises:
        FileValidationError: With a human-readable reason if validation fails.
    """
    # --- Check 1: Does the file exist? ---
    if not os.path.isfile(file_path):
        raise FileValidationError(f"File not found: {file_path}")

    # --- Check 2: Is the extension allowed? ---
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in allowed_extensions:
        allowed_str = ", ".join(allowed_extensions)
        raise FileValidationError(
            f"Invalid file type '{ext}'. Allowed type(s): {allowed_str}"
        )

    # --- Check 3: Is the file empty (0 bytes)? ---
    file_size_bytes = os.path.getsize(file_path)
    if file_size_bytes == 0:
        raise FileValidationError(
            f"The file '{os.path.basename(file_path)}' is empty (0 bytes)."
        )

    # --- Check 4: Is the file too large? ---
    file_size_mb = file_size_bytes / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise FileValidationError(
            f"The file '{os.path.basename(file_path)}' is {file_size_mb:.1f}MB, "
            f"which exceeds the {MAX_FILE_SIZE_MB}MB limit."
        )
