"""
Weather service — fetches forecast data from the free Open-Meteo API and
converts it into crop-disease risk alerts.

Open-Meteo (https://open-meteo.com/) is an open, non-commercial weather API
that does not require an API key. We pull a short 48-hour hourly forecast
and apply simple agronomic heuristics to highlight weather patterns that
commonly drive South Asian crop-disease outbreaks.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
_FORECAST_HOURS = 48


# ---------------------------------------------------------------------------
# Forecast fetch
# ---------------------------------------------------------------------------


async def get_weather_forecast(lat: float, lon: float) -> dict:
    """
    Fetch a 48-hour hourly forecast for the given coordinates.

    Returns the raw Open-Meteo response dict. The caller is expected to pass
    the ``hourly`` block into ``analyze_disease_risk``.

    Raises:
        HTTPException 502: Open-Meteo returned a non-200 status.
        HTTPException 503: Open-Meteo is unreachable (network error).
        HTTPException 504: the request timed out.
    """
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    "temperature_2m,relative_humidity_2m,precipitation_probability",
        "forecast_days": 2,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            response = await client.get(_OPEN_METEO_URL, params=params)
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Weather service timed out. Please try again shortly.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail="Cannot reach the weather service. Check your internet connection.",
        ) from exc

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Weather service returned {response.status_code}: {response.text[:200]}",
        )

    return response.json()


# ---------------------------------------------------------------------------
# Risk analysis
# ---------------------------------------------------------------------------


def _safe_avg(values: list[float | int | None]) -> float:
    """Average over ``values`` ignoring ``None``; returns 0.0 if all None."""
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)


def analyze_disease_risk(weather_data: dict) -> list[dict]:
    """
    Apply simple agronomic heuristics to derive disease risk alerts.

    Looks at the first ``_FORECAST_HOURS`` of the hourly forecast and
    evaluates a small rule-set tailored to common South Asian crop diseases.

    Returns a list of alert dicts with: ``disease``, ``crop``, ``risk``,
    and ``reason``. Multiple rules can match (e.g. humid AND rainy), so the
    list may contain more than one alert.
    """
    hourly = weather_data.get("hourly") or {}

    temps = (hourly.get("temperature_2m")          or [])[:_FORECAST_HOURS]
    hums  = (hourly.get("relative_humidity_2m")    or [])[:_FORECAST_HOURS]
    rains = (hourly.get("precipitation_probability") or [])[:_FORECAST_HOURS]

    avg_temp = _safe_avg(temps)
    avg_hum  = _safe_avg(hums)
    max_rain = max((r for r in rains if r is not None), default=0.0)

    risks: list[dict] = []

    # --- Late Blight — cool + humid ---
    if avg_hum > 80 and 15 <= avg_temp <= 25:
        risks.append({
            "disease": "Late Blight",
            "crop":    "Tomato/Potato",
            "risk":    "HIGH",
            "reason":  "Cool and humid conditions favor Late Blight",
        })

    # --- Bacterial Wilt — warm + humid ---
    if avg_hum > 75 and avg_temp > 25:
        risks.append({
            "disease": "Bacterial Wilt",
            "crop":    "Tomato",
            "risk":    "MEDIUM",
            "reason":  "Warm humid conditions increase bacterial risk",
        })

    # --- Fungal diseases — high rain probability ---
    if max_rain > 70:
        risks.append({
            "disease": "Fungal Diseases",
            "crop":    "All crops",
            "risk":    "MEDIUM",
            "reason":  "High rain probability increases fungal risk",
        })

    # --- Low-risk fallback ---
    if not risks and avg_hum < 60:
        risks.append({
            "disease": "None",
            "crop":    "All",
            "risk":    "LOW",
            "reason":  "Current weather conditions are favorable",
        })

    # --- Neutral fallback so the UI always has something to show ---
    if not risks:
        risks.append({
            "disease": "None",
            "crop":    "All",
            "risk":    "LOW",
            "reason":  "No significant disease-favorable patterns detected",
        })

    return risks
