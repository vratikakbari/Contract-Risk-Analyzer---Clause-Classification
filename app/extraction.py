"""
extraction.py — PDF Text Extraction Module

Uses PyMuPDF (fitz) to read a PDF file and extract text
while preserving page numbers.
"""

import pymupdf
from typing import List, Dict, Union


def extract_text_from_pdf(file_bytes: bytes) -> List[Dict[str, Union[int, str]]]:
    """
    Extract text from a PDF file while preserving page numbers.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        A list of dicts, each containing:
            - page_number (int): 1-indexed page number
            - text (str): Extracted text from that page

    Raises:
        ValueError: If the file cannot be opened as a valid PDF.
    """
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Failed to open PDF: {e}")

    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append({
            "page_number": page_num + 1,  # 1-indexed
            "text": text
        })

    doc.close()
    return pages


def get_full_text(pages: List[Dict[str, Union[int, str]]]) -> str:
    """
    Concatenate all page texts into a single string.

    Args:
        pages: Output from extract_text_from_pdf().

    Returns:
        A single string with all page texts joined by newlines.
    """
    return "\n".join(page["text"] for page in pages)
