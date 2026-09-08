"""
preprocessing.py — Text Cleaning & Normalization Module

Normalizes whitespace, removes non-printable characters,
and strips common PDF extraction artifacts.
"""

import re
import unicodedata


def clean_text(raw_text: str) -> str:
    """
    Clean and normalize extracted PDF text.

    Processing steps:
        1. Remove non-printable / control characters (except newlines and tabs)
        2. Remove form-feed characters (common PDF page-break artifact)
        3. Normalize unicode characters to their canonical form
        4. Replace multiple consecutive blank lines with a single newline
        5. Collapse multiple spaces/tabs on the same line into a single space
        6. Remove common PDF artifacts (repeated header/footer markers, page numbers)
        7. Strip leading and trailing whitespace

    Args:
        raw_text: The raw text extracted from a PDF page or document.

    Returns:
        Cleaned and normalized text string.
    """
    # Step 1: Remove non-printable/control characters (keep \n, \r, \t)
    text = "".join(
        ch for ch in raw_text
        if ch in ("\n", "\r", "\t") or not unicodedata.category(ch).startswith("C")
    )

    # Step 2: Remove form-feed characters
    text = text.replace("\f", "")

    # Step 3: Normalize unicode (e.g., curly quotes → straight quotes)
    text = unicodedata.normalize("NFKD", text)

    # Step 4: Replace multiple consecutive blank lines with a single newline
    text = re.sub(r"\n\s*\n", "\n\n", text)

    # Step 5: Collapse multiple spaces/tabs within a line into a single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Step 6: Remove standalone page-number lines (e.g., "  12  ", "Page 5 of 20")
    text = re.sub(r"(?m)^\s*(?:Page\s+)?\d+(?:\s+of\s+\d+)?\s*$", "", text)

    # Step 7: Remove common artifacts — repeated dashes, underscores, equals signs
    text = re.sub(r"[-_=]{5,}", "", text)

    # Final: Strip leading/trailing whitespace
    text = text.strip()

    return text
