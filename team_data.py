from __future__ import annotations

import pandas as pd


def team_record(team_games: pd.DataFrame, team: str, season: int) -> dict:
    rows = team_games[
        (team_games["team"] == team)
        & (pd.to_numeric(team_games["season"], errors="coerce") == int(season))
    ].copy()

    wins = int(pd.to_numeric(rows.get("win"), errors="coerce").fillna(0).sum()) if not rows.empty else 0
    losses = int(len(rows) - wins)
    return {"wins": wins, "losses": losses, "games": int(len(rows))}


def recent_team_games(team_games: pd.DataFrame, team: str, limit: int = 5) -> pd.DataFrame:
    rows = team_games[team_games["team"] == team].copy()
    if rows.empty:
        return rows

    rows["gameday"] = pd.to_datetime(rows["gameday"], errors="coerce")
    rows = rows.sort_values(["gameday", "game_id"]).tail(limit).copy()
    rows["result"] = rows["win"].map(lambda value: "W" if int(value) == 1 else "L")
    rows["score"] = (
        rows["points_for"].fillna(0).astype(int).astype(str)
        + "–"
        + rows["points_against"].fillna(0).astype(int).astype(str)
    )
    rows["point_diff_display"] = pd.to_numeric(rows["point_diff"], errors="coerce").fillna(0)
    return rows.reset_index(drop=True)


def current_season_summary(team_games: pd.DataFrame, team: str, season: int) -> dict:
    rows = team_games[
        (team_games["team"] == team)
        & (pd.to_numeric(team_games["season"], errors="coerce") == int(season))
    ].copy()

    if rows.empty:
        return {
            "points_for": 0.0,
            "points_against": 0.0,
            "point_diff": 0.0,
            "turnover_margin": 0.0,
        }

    return {
        "points_for": float(pd.to_numeric(rows["points_for"], errors="coerce").mean()),
        "points_against": float(pd.to_numeric(rows["points_against"], errors="coerce").mean()),
        "point_diff": float(pd.to_numeric(rows["point_diff"], errors="coerce").mean()),
        "turnover_margin": float(pd.to_numeric(rows["turnover_margin"], errors="coerce").mean()),
    }


def team_upcoming_games(schedules: pd.DataFrame, team: str, limit: int = 5) -> pd.DataFrame:
    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"] != "PRE"].copy()

    home_score = pd.to_numeric(s.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(s.get("away_score"), errors="coerce")
    future = s[home_score.isna() | away_score.isna()].copy()

    future = future[
        (future["home_team"].astype(str) == team)
        | (future["away_team"].astype(str) == team)
    ].copy()

    if "gameday" in future.columns:
        future["gameday"] = pd.to_datetime(future["gameday"], errors="coerce")
        future = future.sort_values(["gameday", "game_id"])

    return future.head(limit).reset_index(drop=True)
