from __future__ import annotations

import pandas as pd


TEAM_COORDS = {
    "ARI": (33.5276, -112.2626),
    "ATL": (33.7554, -84.4008),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LV": (36.0909, -115.1833),
    "LAC": (33.9535, -118.3392),
    "LA": (33.9535, -118.3392),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9736, -93.2575),
    "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SEA": (47.5952, -122.3316),
    "SF": (37.4030, -121.9700),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9077, -76.8645),
}


def current_week(schedules: pd.DataFrame, season: int) -> int:
    s = schedules.copy()
    s = s[pd.to_numeric(s["season"], errors="coerce") == int(season)].copy()

    if "game_type" in s.columns:
        s = s[s["game_type"].isin(["REG", "POST"])].copy()

    home_score = pd.to_numeric(s.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(s.get("away_score"), errors="coerce")
    future = s[home_score.isna() | away_score.isna()].copy()

    if not future.empty:
        week = pd.to_numeric(future["week"], errors="coerce").dropna()
        if not week.empty:
            return int(week.min())

    completed_week = pd.to_numeric(s["week"], errors="coerce").dropna()
    return int(completed_week.max()) if not completed_week.empty else 1


def games_for_week(
    schedules: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    s = schedules.copy()
    s = s[
        (pd.to_numeric(s["season"], errors="coerce") == int(season))
        & (pd.to_numeric(s["week"], errors="coerce") == int(week))
    ].copy()

    if "game_type" in s.columns:
        s = s[s["game_type"] != "PRE"].copy()

    if "gameday" in s.columns:
        s["gameday"] = pd.to_datetime(s["gameday"], errors="coerce")
        sort_cols = ["gameday"]
        if "gametime" in s.columns:
            sort_cols.append("gametime")
        s = s.sort_values(sort_cols)

    return s.reset_index(drop=True)


def upcoming_games_for_week(
    schedules: pd.DataFrame,
    season: int,
    week: int,
) -> pd.DataFrame:
    games = games_for_week(schedules, season, week)
    if games.empty:
        return games

    home_score = pd.to_numeric(games.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(games.get("away_score"), errors="coerce")
    return games[home_score.isna() | away_score.isna()].reset_index(drop=True)


def league_map_frame(team_names: dict) -> pd.DataFrame:
    rows = []
    for team, (lat, lon) in TEAM_COORDS.items():
        rows.append({
            "team": team,
            "name": team_names.get(team, team),
            "lat": lat,
            "lon": lon,
        })
    return pd.DataFrame(rows)


def regular_season_weeks(schedules: pd.DataFrame, season: int) -> list[int]:
    s = schedules.copy()
    s = s[pd.to_numeric(s["season"], errors="coerce") == int(season)].copy()
    if "game_type" in s.columns:
        s = s[s["game_type"] == "REG"].copy()

    weeks = pd.to_numeric(s.get("week"), errors="coerce").dropna().astype(int)
    return sorted(week for week in weeks.unique().tolist() if week >= 1)


def week_is_complete(schedules: pd.DataFrame, season: int, week: int) -> bool:
    games = games_for_week(schedules, season, week)
    if "game_type" in games.columns:
        games = games[games["game_type"] == "REG"].copy()
    if games.empty:
        return False

    home = pd.to_numeric(games.get("home_score"), errors="coerce")
    away = pd.to_numeric(games.get("away_score"), errors="coerce")
    return bool(home.notna().all() and away.notna().all())


def challenge_active_week(schedules: pd.DataFrame, season: int) -> int:
    weeks = regular_season_weeks(schedules, season)
    if not weeks:
        return 1

    for week in weeks:
        if not week_is_complete(schedules, season, week):
            return week

    return weeks[-1]


def completed_challenge_weeks(schedules: pd.DataFrame, season: int) -> list[int]:
    return [
        week
        for week in regular_season_weeks(schedules, season)
        if week_is_complete(schedules, season, week)
    ]
