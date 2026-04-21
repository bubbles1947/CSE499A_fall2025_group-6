"""
Weather router — exposes a single endpoint that returns current conditions,
a forecast summary, and a list of disease-risk alerts derived from the
Open-Meteo forecast.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from services import weather_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/weather", tags=["Weather"])

# Dhaka, Bangladesh — default when the caller doesn't supply coordinates.
_DEFAULT_LAT = 23.8103
_DEFAULT_LON = 90.4125

# How many hours of the hourly forecast to summarise.
_FORECAST_HOURS = 48


def _safe_first(values: list[float | int | None]) -> float | None:
    """Return the first non-None entry of ``values``, or ``None``."""
    for v in values:
        if v is not None:
            return float(v)
    return None


def _safe_avg(values: list[float | int | None]) -> float:
    clean = [float(v) for v in values if v is not None]
    return round(sum(clean) / len(clean), 1) if clean else 0.0


@router.get(
    "/risk",
    summary="Get current weather and disease-risk forecast for a location",
)
async def get_weather_risk(
    lat: float = Query(_DEFAULT_LAT, description="Latitude (defaults to Dhaka)"),
    lon: float = Query(_DEFAULT_LON, description="Longitude (defaults to Dhaka)"),
) -> dict:
    """
    Fetch a 48-hour hourly forecast from Open-Meteo and convert it into a
    list of crop-disease risk alerts using simple agronomic heuristics.

    Response shape::

        {
          "location":         {"lat": 23.81, "lon": 90.41},
          "current":          {"temperature": 29.5, "humidity": 72.0},
          "forecast_summary": {"avg_temp":   28.7, "avg_humidity": 78.3},
          "risks":            [ {disease, crop, risk, reason}, ... ],
          "updated_at":       "2026-04-20T01:23:45Z"
        }
    """
    try:
        data = await weather_service.get_weather_forecast(lat, lon)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error fetching weather for %s,%s", lat, lon)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while fetching weather data.",
        ) from exc

    hourly = data.get("hourly") or {}
    temps = (hourly.get("temperature_2m")       or [])[:_FORECAST_HOURS]
    hums  = (hourly.get("relative_humidity_2m") or [])[:_FORECAST_HOURS]

    current_temp = _safe_first(temps)
    current_hum  = _safe_first(hums)

    risks = weather_service.analyze_disease_risk(data)

    return {
        "location": {"lat": lat, "lon": lon},
        "current": {
            "temperature": current_temp if current_temp is not None else 0.0,
            "humidity":    current_hum  if current_hum  is not None else 0.0,
        },
        "forecast_summary": {
            "avg_temp":     _safe_avg(temps),
            "avg_humidity": _safe_avg(hums),
        },
        "risks":      risks,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
