from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import requests


INDOOR_ROOFS = {"dome", "closed"}

WEATHER_GAME_COLUMNS = [
    "temperature_f",
    "wind_mph",
    "outdoor_game",
    "weather_known",
    "cold_severity",
    "wind_severity",
    "heat_severity",
]

WEATHER_TEAM_FEATURES = [
    "weather_cold_form",
    "weather_wind_form",
    "weather_heat_form",
]

WEATHER_MODEL_FEATURES = [
    "weather_cold_severity",
    "weather_wind_severity",
    "weather_heat_severity",
    "weather_outdoor",
    "weather_cold_edge",
    "weather_wind_edge",
    "weather_heat_edge",
]

WEATHER_DISPLAY_NAMES = {
    "weather_cold_severity": "Cold-weather severity",
    "weather_wind_severity": "Wind severity",
    "weather_heat_severity": "Heat severity",
    "weather_outdoor": "Outdoor game",
    "weather_cold_edge": "Cold-weather team edge",
    "weather_wind_edge": "Wind-weather team edge",
    "weather_heat_edge": "Hot-weather team edge",
}


# Current home venues. Coordinates are only used for forecast retrieval.
# Historical model training uses nflverse's game-level roof/temp/wind fields.
STADIUMS = {
    "ARI": {"name": "State Farm Stadium", "lat": 33.5276, "lon": -112.2626, "tz": "America/Phoenix", "roof": "closed"},
    "ATL": {"name": "Mercedes-Benz Stadium", "lat": 33.7554, "lon": -84.4008, "tz": "America/New_York", "roof": "closed"},
    "BAL": {"name": "M&T Bank Stadium", "lat": 39.2780, "lon": -76.6227, "tz": "America/New_York", "roof": "outdoors"},
    "BUF": {"name": "Highmark Stadium", "lat": 42.7738, "lon": -78.7870, "tz": "America/New_York", "roof": "outdoors"},
    "CAR": {"name": "Bank of America Stadium", "lat": 35.2258, "lon": -80.8528, "tz": "America/New_York", "roof": "outdoors"},
    "CHI": {"name": "Soldier Field", "lat": 41.8623, "lon": -87.6167, "tz": "America/Chicago", "roof": "outdoors"},
    "CIN": {"name": "Paycor Stadium", "lat": 39.0954, "lon": -84.5160, "tz": "America/New_York", "roof": "outdoors"},
    "CLE": {"name": "Huntington Bank Field", "lat": 41.5061, "lon": -81.6995, "tz": "America/New_York", "roof": "outdoors"},
    "DAL": {"name": "AT&T Stadium", "lat": 32.7473, "lon": -97.0945, "tz": "America/Chicago", "roof": "closed"},
    "DEN": {"name": "Empower Field at Mile High", "lat": 39.7439, "lon": -105.0201, "tz": "America/Denver", "roof": "outdoors"},
    "DET": {"name": "Ford Field", "lat": 42.3400, "lon": -83.0456, "tz": "America/Detroit", "roof": "dome"},
    "GB": {"name": "Lambeau Field", "lat": 44.5013, "lon": -88.0622, "tz": "America/Chicago", "roof": "outdoors"},
    "HOU": {"name": "NRG Stadium", "lat": 29.6847, "lon": -95.4107, "tz": "America/Chicago", "roof": "closed"},
    "IND": {"name": "Lucas Oil Stadium", "lat": 39.7601, "lon": -86.1639, "tz": "America/Indiana/Indianapolis", "roof": "closed"},
    "JAX": {"name": "EverBank Stadium", "lat": 30.3239, "lon": -81.6373, "tz": "America/New_York", "roof": "outdoors"},
    "KC": {"name": "GEHA Field at Arrowhead Stadium", "lat": 39.0489, "lon": -94.4839, "tz": "America/Chicago", "roof": "outdoors"},
    "LV": {"name": "Allegiant Stadium", "lat": 36.0909, "lon": -115.1833, "tz": "America/Los_Angeles", "roof": "dome"},
    "LAC": {"name": "SoFi Stadium", "lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles", "roof": "dome"},
    "LA": {"name": "SoFi Stadium", "lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles", "roof": "dome"},
    "MIA": {"name": "Hard Rock Stadium", "lat": 25.9580, "lon": -80.2389, "tz": "America/New_York", "roof": "outdoors"},
    "MIN": {"name": "U.S. Bank Stadium", "lat": 44.9736, "lon": -93.2575, "tz": "America/Chicago", "roof": "dome"},
    "NE": {"name": "Gillette Stadium", "lat": 42.0909, "lon": -71.2643, "tz": "America/New_York", "roof": "outdoors"},
    "NO": {"name": "Caesars Superdome", "lat": 29.9511, "lon": -90.0812, "tz": "America/Chicago", "roof": "dome"},
    "NYG": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745, "tz": "America/New_York", "roof": "outdoors"},
    "NYJ": {"name": "MetLife Stadium", "lat": 40.8135, "lon": -74.0745, "tz": "America/New_York", "roof": "outdoors"},
    "PHI": {"name": "Lincoln Financial Field", "lat": 39.9008, "lon": -75.1675, "tz": "America/New_York", "roof": "outdoors"},
    "PIT": {"name": "Acrisure Stadium", "lat": 40.4468, "lon": -80.0158, "tz": "America/New_York", "roof": "outdoors"},
    "SEA": {"name": "Lumen Field", "lat": 47.5952, "lon": -122.3316, "tz": "America/Los_Angeles", "roof": "outdoors"},
    "SF": {"name": "Levi's Stadium", "lat": 37.4030, "lon": -121.9700, "tz": "America/Los_Angeles", "roof": "outdoors"},
    "TB": {"name": "Raymond James Stadium", "lat": 27.9759, "lon": -82.5033, "tz": "America/New_York", "roof": "outdoors"},
    "TEN": {"name": "Nissan Stadium", "lat": 36.1665, "lon": -86.7713, "tz": "America/Chicago", "roof": "outdoors"},
    "WAS": {"name": "Northwest Stadium", "lat": 38.9077, "lon": -76.8645, "tz": "America/New_York", "roof": "outdoors"},
}


def _numeric_series(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def add_schedule_weather(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize nflverse historical weather into model-friendly severity values."""
    out = frame.copy()

    roof = (
        out["roof"].fillna("").astype(str).str.lower()
        if "roof" in out.columns
        else pd.Series("", index=out.index, dtype=str)
    )
    raw_temp = _numeric_series(out, "temp")
    raw_wind = _numeric_series(out, "wind")

    indoor = roof.isin(INDOOR_ROOFS)
    known = indoor | raw_temp.notna() | raw_wind.notna()

    temp = raw_temp.copy()
    wind = raw_wind.copy()

    # Indoor weather is neutral. Missing outdoor observations also receive
    # neutral values, while weather_known=0 prevents them from creating edges.
    temp = temp.where(~indoor, 70.0).fillna(65.0)
    wind = wind.where(~indoor, 0.0).fillna(5.0)

    outdoor = (~indoor).astype(float)
    known_float = known.astype(float)

    out["temperature_f"] = temp.astype(float)
    out["wind_mph"] = wind.astype(float)
    out["outdoor_game"] = outdoor
    out["weather_known"] = known_float
    out["cold_severity"] = (((40.0 - temp).clip(lower=0.0) / 20.0).clip(upper=2.5) * outdoor * known_float)
    out["wind_severity"] = (((wind - 10.0).clip(lower=0.0) / 15.0).clip(upper=3.0) * outdoor * known_float)
    out["heat_severity"] = (((temp - 80.0).clip(lower=0.0) / 20.0).clip(upper=2.0) * outdoor * known_float)
    return out


def _prior_condition_form(group: pd.DataFrame, severity_column: str) -> pd.Series:
    """Prior point-differential performance in a condition, shrunk to overall form."""
    results = []
    overall_sum = 0.0
    overall_count = 0
    condition_sum = 0.0
    condition_count = 0
    prior_strength = 3.0

    for row in group.itertuples():
        overall_mean = overall_sum / overall_count if overall_count else 0.0

        if condition_count:
            condition_mean = (
                condition_sum + prior_strength * overall_mean
            ) / (condition_count + prior_strength)
        else:
            condition_mean = overall_mean

        results.append(float(condition_mean))

        point_diff = float(getattr(row, "point_diff", 0.0))
        severity = float(getattr(row, severity_column, 0.0))
        overall_sum += point_diff
        overall_count += 1

        if severity > 0:
            condition_sum += point_diff
            condition_count += 1

    return pd.Series(results, index=group.index, dtype=float)


def add_team_weather_history(team_games: pd.DataFrame) -> pd.DataFrame:
    out = team_games.copy()
    mapping = {
        "cold_severity": "weather_cold_form",
        "wind_severity": "weather_wind_form",
        "heat_severity": "weather_heat_form",
    }

    for target in mapping.values():
        out[target] = 0.0

    for _, group in out.groupby("team", sort=False):
        group = group.sort_values(["gameday", "game_id"])
        for severity, target in mapping.items():
            out.loc[group.index, target] = _prior_condition_form(group, severity)

    return out


def model_weather_values(weather: dict | None, home_snapshot: dict, away_snapshot: dict) -> dict:
    weather = weather or {}
    confidence = float(weather.get("confidence_weight", 1.0))
    known = float(weather.get("weather_known", 0.0))

    cold = float(weather.get("cold_severity", 0.0)) * confidence * known
    wind = float(weather.get("wind_severity", 0.0)) * confidence * known
    heat = float(weather.get("heat_severity", 0.0)) * confidence * known
    outdoor = float(weather.get("outdoor_game", 0.0)) * known

    return {
        "weather_cold_severity": cold,
        "weather_wind_severity": wind,
        "weather_heat_severity": heat,
        "weather_outdoor": outdoor,
        "weather_cold_edge": cold * (
            float(home_snapshot.get("weather_cold_form", 0.0))
            - float(away_snapshot.get("weather_cold_form", 0.0))
        ),
        "weather_wind_edge": wind * (
            float(home_snapshot.get("weather_wind_form", 0.0))
            - float(away_snapshot.get("weather_wind_form", 0.0))
        ),
        "weather_heat_edge": heat * (
            float(home_snapshot.get("weather_heat_form", 0.0))
            - float(away_snapshot.get("weather_heat_form", 0.0))
        ),
    }


def _neutral_forecast(reason: str, **extra) -> dict:
    result = {
        "available": False,
        "reason": reason,
        "weather_known": 0.0,
        "outdoor_game": 0.0,
        "cold_severity": 0.0,
        "wind_severity": 0.0,
        "heat_severity": 0.0,
        "confidence_weight": 0.0,
    }
    result.update(extra)
    return result


def _condition_from_code(code, precipitation: float, temp_f: float) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        code = -1

    if code in {71, 73, 75, 77, 85, 86}:
        return "Snow"
    if code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}:
        return "Rain"
    if code in {95, 96, 99}:
        return "Thunderstorms"
    if precipitation > 0 and temp_f <= 34:
        return "Wintry precipitation"
    if precipitation > 0:
        return "Precipitation"
    if code in {0, 1}:
        return "Clear"
    if code in {2, 3, 45, 48}:
        return "Cloudy"
    return "Forecast available"


def _confidence(hours_until: float) -> tuple[float, str]:
    if hours_until <= 3:
        return 1.00, "Very high"
    if hours_until <= 12:
        return 0.93, "Very high"
    if hours_until <= 24:
        return 0.85, "High"
    if hours_until <= 48:
        return 0.75, "Moderate-high"
    return 0.65, "Moderate"


def _matchup_row(schedules: pd.DataFrame, away_team: str, home_team: str):
    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"] != "PRE"].copy()

    home_score = pd.to_numeric(s.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(s.get("away_score"), errors="coerce")
    future = s[home_score.isna() | away_score.isna()].copy()
    future = future[
        (future["away_team"].astype(str) == away_team)
        & (future["home_team"].astype(str) == home_team)
    ].copy()

    if future.empty:
        return None

    future["gameday_sort"] = pd.to_datetime(future["gameday"], errors="coerce")
    future = future.sort_values(["gameday_sort", "game_id"])
    return future.iloc[0]


def forecast_for_matchup(schedules: pd.DataFrame, away_team: str, home_team: str) -> dict:
    row = _matchup_row(schedules, away_team, home_team)
    if row is None:
        return _neutral_forecast("No upcoming scheduled matchup was found for these teams.")

    gameday = pd.to_datetime(row.get("gameday"), errors="coerce")
    if pd.isna(gameday):
        return _neutral_forecast("The upcoming game date is unavailable.")

    gametime_text = str(row.get("gametime") or "13:00")
    try:
        hour, minute = [int(x) for x in gametime_text.split(":")[:2]]
    except Exception:
        hour, minute = 13, 0

    kickoff_et = datetime.combine(
        gameday.date(),
        time(hour=hour, minute=minute),
        tzinfo=ZoneInfo("America/New_York"),
    )

    stadium = STADIUMS.get(home_team)
    if not stadium:
        return _neutral_forecast(
            "No forecast location is configured for this home team.",
            kickoff=kickoff_et.isoformat(),
        )

    location_value = str(row.get("location") or "").lower()
    if location_value == "neutral":
        return _neutral_forecast(
            "Neutral-site weather is not enabled yet because the home-team stadium may be incorrect.",
            kickoff=kickoff_et.isoformat(),
        )

    local_tz = ZoneInfo(stadium["tz"])
    kickoff_local = kickoff_et.astimezone(local_tz)
    now_utc = datetime.now(ZoneInfo("UTC"))
    hours_until = (kickoff_et.astimezone(ZoneInfo("UTC")) - now_utc).total_seconds() / 3600.0

    roof = str(row.get("roof") or stadium["roof"]).lower()
    indoor = roof in INDOOR_ROOFS

    base = {
        "kickoff": kickoff_local.isoformat(),
        "stadium": stadium["name"],
        "roof": roof,
    }

    if hours_until > 72:
        return _neutral_forecast(
            "Weather monitoring begins 72 hours before kickoff.",
            hours_until=hours_until,
            **base,
        )

    if hours_until < -6:
        return _neutral_forecast("This scheduled game has already started or passed.", **base)

    if indoor:
        return {
            "available": True,
            "indoor": True,
            "condition": "Indoor / roof closed",
            "temperature_f": 70.0,
            "wind_mph": 0.0,
            "wind_gust_mph": 0.0,
            "precip_probability": 0.0,
            "precipitation_in": 0.0,
            "weather_known": 1.0,
            "outdoor_game": 0.0,
            "cold_severity": 0.0,
            "wind_severity": 0.0,
            "heat_severity": 0.0,
            "confidence_weight": 1.0,
            "confidence_label": "Indoor",
            "hours_until": hours_until,
            **base,
        }

    confidence_weight, confidence_label = _confidence(max(hours_until, 0.0))

    params = {
        "latitude": stadium["lat"],
        "longitude": stadium["lon"],
        "hourly": ",".join([
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "weather_code",
        ]),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": stadium["tz"],
        "start_date": kickoff_local.date().isoformat(),
        "end_date": kickoff_local.date().isoformat(),
    }

    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return _neutral_forecast("The forecast service returned no hourly data.", **base)

        target = kickoff_local.replace(tzinfo=None)
        parsed = [datetime.fromisoformat(value) for value in times]
        index = min(range(len(parsed)), key=lambda i: abs((parsed[i] - target).total_seconds()))

        def value(name, default=0.0):
            values = hourly.get(name, [])
            if index >= len(values) or values[index] is None:
                return default
            return values[index]

        temp_f = float(value("temperature_2m", 65.0))
        wind_mph = float(value("wind_speed_10m", 5.0))
        gust_mph = float(value("wind_gusts_10m", wind_mph))
        precip_probability = float(value("precipitation_probability", 0.0))
        precipitation = float(value("precipitation", 0.0))
        weather_code = value("weather_code", -1)

        cold = min(max((40.0 - temp_f) / 20.0, 0.0), 2.5)
        windy = min(max((wind_mph - 10.0) / 15.0, 0.0), 3.0)
        heat = min(max((temp_f - 80.0) / 20.0, 0.0), 2.0)

        return {
            "available": True,
            "indoor": False,
            "condition": _condition_from_code(weather_code, precipitation, temp_f),
            "temperature_f": temp_f,
            "wind_mph": wind_mph,
            "wind_gust_mph": gust_mph,
            "precip_probability": precip_probability,
            "precipitation_in": precipitation,
            "weather_known": 1.0,
            "outdoor_game": 1.0,
            "cold_severity": cold,
            "wind_severity": windy,
            "heat_severity": heat,
            "confidence_weight": confidence_weight,
            "confidence_label": confidence_label,
            "hours_until": hours_until,
            **base,
        }

    except (requests.RequestException, ValueError, TypeError) as exc:
        return _neutral_forecast(
            f"Forecast temporarily unavailable: {exc}",
            hours_until=hours_until,
            **base,
        )
