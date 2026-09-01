from __future__ import annotations

import pandas as pd
import nflreadpy as nfl


TEAM_ALIASES = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV",
    "OAK": "LV", "SDG": "LAC", "STL": "LA", "WSH": "WAS",
}


def _to_pandas(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _norm_team(value):
    if pd.isna(value):
        return value
    value = str(value).upper().strip()
    return TEAM_ALIASES.get(value, value)


def load_injury_data(season: int) -> pd.DataFrame:
    try:
        injuries = _to_pandas(nfl.load_injuries([int(season)]))
    except Exception:
        return pd.DataFrame()

    if injuries.empty:
        return injuries

    if "team" in injuries.columns:
        injuries["team"] = injuries["team"].map(_norm_team)

    if "season_type" in injuries.columns:
        injuries = injuries[injuries["season_type"].isin(["REG", "POST"])].copy()

    if "week" in injuries.columns:
        injuries["week"] = pd.to_numeric(injuries["week"], errors="coerce")

    if "date_modified" in injuries.columns:
        injuries["date_modified"] = pd.to_datetime(
            injuries["date_modified"],
            errors="coerce",
            utc=True,
        )

    return injuries


def latest_team_injuries(injuries: pd.DataFrame, team: str) -> pd.DataFrame:
    if injuries.empty or "team" not in injuries.columns:
        return pd.DataFrame()

    team_rows = injuries[injuries["team"] == team].copy()
    if team_rows.empty:
        return team_rows

    if "week" in team_rows.columns and team_rows["week"].notna().any():
        latest_week = team_rows["week"].max()
        team_rows = team_rows[team_rows["week"] == latest_week].copy()

    sort_cols = []
    if "date_modified" in team_rows.columns:
        sort_cols.append("date_modified")
    if sort_cols:
        team_rows = team_rows.sort_values(sort_cols, ascending=False)

    player_key = "gsis_id" if "gsis_id" in team_rows.columns else "full_name"
    if player_key in team_rows.columns:
        team_rows = team_rows.drop_duplicates(player_key, keep="first")

    keep = [
        c for c in [
            "week",
            "full_name",
            "position",
            "report_primary_injury",
            "report_secondary_injury",
            "report_status",
            "practice_status",
            "date_modified",
        ]
        if c in team_rows.columns
    ]
    return team_rows[keep].reset_index(drop=True)


def injury_status_counts(team_injuries: pd.DataFrame) -> dict:
    if team_injuries.empty or "report_status" not in team_injuries.columns:
        return {"Out": 0, "Doubtful": 0, "Questionable": 0}

    statuses = team_injuries["report_status"].fillna("").astype(str).str.lower().str.strip()
    return {
        "Out": int((statuses == "out").sum()),
        "Doubtful": int((statuses == "doubtful").sum()),
        "Questionable": int((statuses == "questionable").sum()),
    }
