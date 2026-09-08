"""
domain_screening.py — Document Domain Classification & Contract Type Classification

Uses a zero-shot Hugging Face classifier (typeform/distilbert-base-uncased-mnli) to:
  1. Validate whether the document belongs to a supported legal domain.
  2. Classify the specific contract type within the accepted domain.
"""

from transformers import pipeline
from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "facebook/bart-large-mnli"


# ---------------------------------------------------------------------------
# Singleton model loader — loaded once, shared across domain & clause modules
# ---------------------------------------------------------------------------

_classifier = None


def get_classifier():
    """
    Return a cached zero-shot-classification pipeline.
    The model is loaded on first call and reused afterwards.
    """
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,  # CPU; set to 0 for GPU
        )
    return _classifier


# ---------------------------------------------------------------------------
# Domain labels and contract-type labels per domain
# ---------------------------------------------------------------------------

DOMAIN_LABELS = [
    "Business and Commercial Contract Law",
    "Real Estate and Property Law",
    "Construction and Engineering Law",
]

CONTRACT_TYPE_LABELS: Dict[str, list] = {
    "Business and Commercial Contract Law": [
        "Service Agreement",
        "Employment Contract",
        "Non-Disclosure Agreement (NDA)",
        "Partnership Agreement",
        "Licensing Agreement",
    ],
    "Real Estate and Property Law": [
        "Sale Agreement",
        "Lease Agreement",
        "Rental Agreement",
        "Mortgage Agreement",
        "Property Management Agreement",
    ],
    "Construction and Engineering Law": [
        "Construction Contract",
        "Subcontractor Agreement",
        "Design-Build Contract",
        "Engineering Procurement Construction (EPC) Contract",
    ],
}

# Minimum confidence to accept a document into a legal domain
DOMAIN_CONFIDENCE_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def screen_domain(text: str, threshold: float = DOMAIN_CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    """
    Classify the document into one of the supported legal domains.

    Args:
        text:      The cleaned full-text of the document (truncated internally
                   to the first ~1024 tokens worth of characters for speed).
        threshold: Minimum confidence for the top domain label to accept.

    Returns:
        dict with keys:
            - status: "Accepted" or "Rejected"
            - domain: the predicted domain label (only if accepted)
            - domain_confidence: float confidence score
            - reason: explanation string (only if rejected)
    """
    classifier = get_classifier()

    # Use the first ~3000 characters (roughly 700-800 tokens) for speed
    sample = text[:3000]

    result = classifier(sample, candidate_labels=DOMAIN_LABELS)
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    if top_score < threshold:
        return {
            "status": "Rejected",
            "domain": top_label,
            "domain_confidence": round(top_score, 4),
            "reason": "Document does not belong to a supported legal domain.",
        }

    return {
        "status": "Accepted",
        "domain": top_label,
        "domain_confidence": round(top_score, 4),
    }


def classify_contract_type(text: str, domain: str) -> Dict[str, Any]:
    """
    Classify the specific contract type within the accepted domain.

    Args:
        text:   The cleaned full-text of the document.
        domain: The domain label returned by screen_domain().

    Returns:
        dict with keys:
            - contract_type: predicted contract type string
            - contract_type_confidence: float confidence score
    """
    classifier = get_classifier()
    candidate_labels = CONTRACT_TYPE_LABELS.get(domain, [])

    if not candidate_labels:
        return {
            "contract_type": "Unknown",
            "contract_type_confidence": 0.0,
        }

    # Use the first ~3000 characters for classification
    sample = text[:3000]

    result = classifier(sample, candidate_labels=candidate_labels)
    return {
        "contract_type": result["labels"][0],
        "contract_type_confidence": round(result["scores"][0], 4),
    }
