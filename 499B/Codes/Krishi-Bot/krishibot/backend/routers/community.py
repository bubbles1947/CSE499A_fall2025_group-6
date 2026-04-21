"""
Community disease board — anonymous crop-disease reports submitted by
farmers so others can see what's spreading in their region.

No user identity, photos, or personal information are stored — just the
disease, crop, location, and severity. The board is read-only for viewers
and write-only for submitters; there is no authentication layer.
"""

from __future__ import annotations

import logging
import random
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from models.database import CommunityReport
from schemas.schemas import (
    CommunityReportCreate,
    CommunityReportResponse,
    CommunityStatsResponse,
    DiseaseCount,
    DivisionCount,
    LocationCount,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/community", tags=["Community"])


# ---------------------------------------------------------------------------
# Static catalogues
# ---------------------------------------------------------------------------


BANGLADESH_DIVISIONS: list[str] = [
    "Dhaka",
    "Chittagong",
    "Rajshahi",
    "Khulna",
    "Barisal",
    "Sylhet",
    "Rangpur",
    "Mymensingh",
]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/divisions",
    response_model=list[str],
    summary="List Bangladesh divisions",
)
async def list_divisions() -> list[str]:
    """Return the eight administrative divisions of Bangladesh."""
    return BANGLADESH_DIVISIONS


@router.post(
    "/report",
    response_model=CommunityReportResponse,
    status_code=201,
    summary="Submit an anonymous disease report",
)
async def create_report(
    body: CommunityReportCreate,
    db: Session = Depends(get_db),
) -> CommunityReport:
    """
    Persist a new anonymous community report.

    The caller is expected to have validated input via Pydantic; this handler
    only enforces that divisions stay within the supported list (defaulting
    silently if an unknown value is supplied).
    """
    division = body.division if body.division in BANGLADESH_DIVISIONS else "Dhaka"

    report = CommunityReport(
        disease_name  = body.disease_name.strip(),
        crop_name     = body.crop_name.strip(),
        location_name = body.location_name.strip(),
        division      = division,
        severity      = body.severity,
        notes         = (body.notes or "").strip() or None,
        is_anonymous  = True,
        report_date   = date.today().isoformat(),
    )

    db.add(report)
    db.commit()
    db.refresh(report)
    return report


@router.get(
    "/reports",
    response_model=list[CommunityReportResponse],
    summary="List recent community reports",
)
async def list_reports(
    division: str | None = Query(default=None, description="Filter by division"),
    crop:     str | None = Query(default=None, description="Filter by crop name"),
    limit:    int         = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> list[CommunityReport]:
    """Return the most recent reports, optionally filtered by division or crop."""
    query = db.query(CommunityReport)

    if division:
        query = query.filter(CommunityReport.division == division)
    if crop:
        query = query.filter(func.lower(CommunityReport.crop_name) == crop.lower())

    return (
        query.order_by(CommunityReport.created_at.desc())
             .limit(limit)
             .all()
    )


@router.get(
    "/stats",
    response_model=CommunityStatsResponse,
    summary="Aggregate stats for the community board",
)
async def get_stats(db: Session = Depends(get_db)) -> CommunityStatsResponse:
    """
    Return headline numbers for the community board:
    total reports, top diseases, top locations, recent reports, and
    per-division counts.
    """
    total = db.query(CommunityReport).count()

    top_diseases_rows = (
        db.query(CommunityReport.disease_name, func.count(CommunityReport.id))
          .group_by(CommunityReport.disease_name)
          .order_by(func.count(CommunityReport.id).desc())
          .limit(5)
          .all()
    )

    top_locations_rows = (
        db.query(CommunityReport.location_name, func.count(CommunityReport.id))
          .group_by(CommunityReport.location_name)
          .order_by(func.count(CommunityReport.id).desc())
          .limit(5)
          .all()
    )

    division_rows = (
        db.query(CommunityReport.division, func.count(CommunityReport.id))
          .filter(CommunityReport.division.isnot(None))
          .group_by(CommunityReport.division)
          .order_by(func.count(CommunityReport.id).desc())
          .all()
    )

    recent = (
        db.query(CommunityReport)
          .order_by(CommunityReport.created_at.desc())
          .limit(10)
          .all()
    )

    return CommunityStatsResponse(
        total_reports       = total,
        top_diseases        = [DiseaseCount(disease_name=d, count=c)  for d, c in top_diseases_rows],
        top_locations       = [LocationCount(location_name=l, count=c) for l, c in top_locations_rows],
        reports_by_division = [DivisionCount(division=d, count=c)      for d, c in division_rows],
        recent_reports      = [CommunityReportResponse.model_validate(r) for r in recent],
    )


# ---------------------------------------------------------------------------
# Seed (development only)
# ---------------------------------------------------------------------------


# Pool of realistic entries for the sample data.
_SAMPLE_DISEASES: list[tuple[str, str, str]] = [
    ("Late Blight",     "Tomato",  "High"),
    ("Late Blight",     "Potato",  "High"),
    ("Early Blight",    "Tomato",  "Medium"),
    ("Early Blight",    "Potato",  "Medium"),
    ("Bacterial Wilt",  "Tomato",  "High"),
    ("Leaf Spot",       "Rice",    "Medium"),
    ("Brown Spot",      "Rice",    "Low"),
    ("Rice Blast",      "Rice",    "High"),
    ("Mosaic Virus",    "Tomato",  "Medium"),
    ("Mosaic Virus",    "Potato",  "Medium"),
]

_SAMPLE_LOCATIONS: list[tuple[str, str]] = [
    ("Gazipur",      "Dhaka"),
    ("Savar",        "Dhaka"),
    ("Comilla",      "Chittagong"),
    ("Feni",         "Chittagong"),
    ("Natore",       "Rajshahi"),
    ("Bogra",        "Rajshahi"),
    ("Jessore",      "Khulna"),
    ("Satkhira",     "Khulna"),
    ("Barisal Sadar","Barisal"),
    ("Sylhet Sadar", "Sylhet"),
    ("Moulvibazar",  "Sylhet"),
    ("Rangpur Sadar","Rangpur"),
    ("Dinajpur",     "Rangpur"),
    ("Mymensingh Sadar", "Mymensingh"),
    ("Jamalpur",     "Mymensingh"),
]

_SAMPLE_NOTES: list[str | None] = [
    "Spreading fast in the neighbourhood.",
    "Noticed after last week's rain.",
    "Only a few plants affected so far.",
    "Third season in a row we see this.",
    None,
    None,
    "Applied copper fungicide — seeing some improvement.",
    "Lost about 20% of the field.",
    None,
    "Worse on the older leaves.",
]


def seed_sample_data(db: Session, count: int = 15) -> int:
    """
    Insert a batch of realistic Bangladesh-style sample reports.

    Intended for development/demo use only. Entries are distributed across
    the last 30 days so "Recent reports" and "Top diseases this week" both
    have content to render.

    Returns the number of rows inserted.
    """
    rng = random.Random(42)  # deterministic seed so demos are reproducible

    now = datetime.now(timezone.utc)
    inserted = 0

    for _ in range(count):
        disease, crop, severity = rng.choice(_SAMPLE_DISEASES)
        location, division      = rng.choice(_SAMPLE_LOCATIONS)
        notes                   = rng.choice(_SAMPLE_NOTES)

        days_ago  = rng.randint(0, 29)
        hours_ago = rng.randint(0, 23)
        created   = now - timedelta(days=days_ago, hours=hours_ago)

        db.add(CommunityReport(
            disease_name  = disease,
            crop_name     = crop,
            location_name = location,
            division      = division,
            severity      = severity,
            notes         = notes,
            is_anonymous  = True,
            created_at    = created,
            report_date   = created.date().isoformat(),
        ))
        inserted += 1

    db.commit()
    return inserted


@router.post(
    "/seed",
    summary="Insert sample community reports (development only)",
)
async def seed_endpoint(db: Session = Depends(get_db)) -> dict:
    """
    Populate the community board with 15 realistic sample reports.

    Skips silently if reports already exist so repeated calls during a demo
    don't balloon the table.
    """
    existing = db.query(CommunityReport).count()
    if existing > 0:
        return {
            "status":   "skipped",
            "message":  f"Community board already has {existing} report(s); seed aborted.",
            "inserted": 0,
        }

    try:
        inserted = seed_sample_data(db, count=15)
    except Exception as exc:
        logger.exception("Failed to seed community reports")
        raise HTTPException(
            status_code=500,
            detail="Failed to seed community reports.",
        ) from exc

    return {"status": "ok", "message": "Seed data inserted.", "inserted": inserted}
