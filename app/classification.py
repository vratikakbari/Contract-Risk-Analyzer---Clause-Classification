"""
classification.py — Clause Classification Module

Uses a zero-shot Hugging Face transformer (facebook/bart-large-mnli)
to categorize each clause into one of the predefined legal labels.
"""

from typing import List, Dict, Any
from app.domain_screening import get_classifier


# ---------------------------------------------------------------------------
# Clause category labels
# ---------------------------------------------------------------------------

CLAUSE_LABELS = [
    "Parties",
    "Effective Date",
    "Termination",
    "Payment",
    "Penalties",
    "Indemnification",
    "Limitation of Liability",
    "Governing Law",
    "Dispute Resolution",
    "Confidentiality",
]


def classify_clauses(
    clauses: List[Dict[str, str]],
    labels: List[str] = CLAUSE_LABELS,
    max_chars: int = 1500,
) -> List[Dict[str, Any]]:
    """
    Classify each clause into one of the predefined legal categories.

    Uses the same zero-shot classifier singleton loaded in domain_screening
    to avoid duplicate model loading.

    Args:
        clauses:   List of clause dicts with keys 'clause_id' and 'text'
                   (output of segmentation.segment_clauses).
        labels:    Candidate label strings for classification.
        max_chars: Maximum characters per clause to send to the model.
                   Longer clauses are truncated.

    Returns:
        The same list of clause dicts, each enriched with:
            - label (str): Top predicted clause category
            - confidence (float): Confidence score for the top label
    """
    classifier = get_classifier()
    classified: List[Dict[str, Any]] = []

    for clause in clauses:
        text = clause["text"][:max_chars]  # Truncate for model context window

        if not text.strip():
            classified.append({
                **clause,
                "label": "Unknown",
                "confidence": 0.0,
            })
            continue

        result = classifier(text, candidate_labels=labels)
        classified.append({
            **clause,
            "label": result["labels"][0],
            "confidence": round(result["scores"][0], 4),
        })

    return classified
