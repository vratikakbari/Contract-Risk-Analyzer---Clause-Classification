"""
segmentation.py — Clause Segmentation Module

Splits cleaned legal text into individual clauses based on
standard legal numbering patterns using regex.
"""

import re
from typing import List, Dict


# ---------------------------------------------------------------------------
# Regex pattern for standard legal clause numbering
# ---------------------------------------------------------------------------
# Matches patterns like:
#   1.          1.1        1.1.1       (a)        (i)        (1)
#   ARTICLE I   ARTICLE 1  Section 1   CLAUSE 1
#   Article I.  Section 1.1

CLAUSE_PATTERN = re.compile(
    r"""
    (?:^|\n)                                  # Start of string or newline
    \s*                                       # Optional leading whitespace
    (?:
        (?:ARTICLE|Article|SECTION|Section|CLAUSE|Clause)  # Named headings
        \s+
        (?:[IVXLCDM]+|\d+)                   # Roman numeral or digit
        (?:\.\d+)*                            # Optional sub-numbering (.1, .1.1)
        [.:\s]                                # Followed by period, colon, or space
    |
        \d+                                   # Numeric: 1, 12, etc.
        (?:\.\d+)*                            # Optional sub-numbering: 1.1, 1.1.1
        [.)\s]                                # Followed by period, paren, or space
    |
        \([a-z]\)                             # Lettered: (a), (b), ...
    |
        \([ivxlcdm]+\)                        # Roman-numeral lettered: (i), (ii), ...
    |
        \(\d+\)                               # Numbered in parentheses: (1), (2), ...
    |
        [A-Z]\.\s                             # Single capital letter: A. , B. , ...
    )
    """,
    re.VERBOSE | re.MULTILINE,
)


def segment_clauses(text: str, min_length: int = 20) -> List[Dict[str, str]]:
    """
    Split cleaned legal text into clause segments based on numbering patterns.

    Args:
        text:       Cleaned full-text of the document.
        min_length: Minimum character length for a clause. Shorter fragments
                    are merged into the preceding clause to reduce noise.

    Returns:
        A list of dicts, each containing:
            - clause_id (str): The matched numbering label (e.g., "1.1", "(a)")
            - text (str): The body text of that clause
    """
    # Find all clause-start positions
    matches = list(CLAUSE_PATTERN.finditer(text))

    if not matches:
        # No clause boundaries detected — return entire text as one block
        return [{"clause_id": "1", "text": text.strip()}]

    clauses: List[Dict[str, str]] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Extract the clause identifier from the matched portion
        clause_header = match.group().strip()
        clause_body = text[match.end():end].strip()

        # Clean up the clause_id — remove trailing punctuation
        clause_id = re.sub(r"[.:)\s]+$", "", clause_header)
        clause_id = clause_id.strip()

        if not clause_id:
            clause_id = f"clause_{i + 1}"

        clauses.append({
            "clause_id": clause_id,
            "text": clause_body,
        })

    # Merge short fragments into the preceding clause
    merged: List[Dict[str, str]] = []
    for clause in clauses:
        if merged and len(clause["text"]) < min_length:
            merged[-1]["text"] += " " + clause["text"]
        else:
            merged.append(clause)

    return merged
