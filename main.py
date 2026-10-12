"""
main.py — FastAPI Application Entry Point

Exposes a POST /analyze endpoint that accepts a PDF upload,
validates the document domain, and returns structured JSON
with contract type and classified clauses.

The zero-shot classification model is loaded once at startup
via FastAPI's lifespan event to avoid first-request latency.
"""

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.extraction import extract_text_from_pdf, get_full_text
from app.preprocessing import clean_text
from app.domain_screening import screen_domain, classify_contract_type, get_classifier
from app.segmentation import segment_clauses
from app.classification import classify_clauses

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Lifespan — pre-load the ML model at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the zero-shot classifier so the first request is fast."""
    logger.info("Loading zero-shot classification model (facebook/bart-large-mnli)...")
    get_classifier()  # triggers singleton creation & caching
    logger.info("Model loaded successfully. Server is ready.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Legal Contract Analysis API",
    description=(
        "AI-powered document intelligence pipeline that extracts text from "
        "legal PDF contracts, validates the document domain, classifies the "
        "contract type, and categorizes individual clauses."
    ),
    version="1.0.0",
    lifespan=lifespan,
    swagger_ui_parameters={"tryItOutEnabled": True},
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
async def root():
    """Health-check endpoint."""
    return {"status": "ok", "message": "Legal Contract Analysis API is running."}


# ---------------------------------------------------------------------------
# Main analysis endpoint
# ---------------------------------------------------------------------------

@app.post("/analyze", tags=["analysis"])
async def analyze_contract(
    file: Annotated[
        UploadFile,
        File(..., description="Upload a PDF contract file", media_type="application/pdf"),
    ]
):
    """
    Upload a PDF legal contract for full analysis.

    Pipeline:
        1. Validate file type (must be PDF)
        2. Extract text from PDF (preserving page numbers)
        3. Clean and normalize the extracted text
        4. Screen the document domain (Legal, Real Estate, Construction)
        5. If rejected, return rejection response immediately
        6. Classify the specific contract type within the domain
        7. Segment the text into individual clauses
        8. Classify each clause into legal categories
        9. Return structured JSON response

    Returns:
        JSON with status, domain, contract_type, and classified clauses.
    """

    # ---- Step 1: Validate file type ----
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(
            status_code = 400,
            detail={
                "error": "Invalid file type.",
                "expected": "application/pdf",
                "received": file.content_type,
            },
        )

    # ---- Step 2: Extract text ----
    try:
        file_bytes = await file.read()
        pages = extract_text_from_pdf(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)})

    full_text = get_full_text(pages)

    if not full_text.strip():
        raise HTTPException(
            status_code=400,
            detail={"error": "No extractable text found in the PDF."},
        )

    # ---- Step 3: Clean text ----
    cleaned_text = clean_text(full_text)

    # ---- Step 4: Domain screening ----
    domain_result = screen_domain(cleaned_text)

    if domain_result["status"] == "Rejected":
        return JSONResponse(
            status_code=200,
            content={
                "status": "Rejected",
                "domain": domain_result.get("domain"),
                "domain_confidence": domain_result.get("domain_confidence"),
                "reason": domain_result["reason"],
            },
        )

    # ---- Step 5: Contract-type classification ----
    contract_result = classify_contract_type(
        cleaned_text, domain_result["domain"]
    )

    # ---- Step 6: Clause segmentation ----
    clauses = segment_clauses(cleaned_text)

    # ---- Step 7: Clause classification ----
    classified_clauses = classify_clauses(clauses)

    # ---- Step 8: Build response ----
    return {
        "status": "Accepted",
        "domain": domain_result["domain"],
        "domain_confidence": domain_result["domain_confidence"],
        "contract_type": contract_result["contract_type"],
        "contract_type_confidence": contract_result["contract_type_confidence"],
        "total_pages": len(pages),
        "total_clauses": len(classified_clauses),
        "clauses": classified_clauses,
    }
