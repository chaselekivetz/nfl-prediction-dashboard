from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd
import nflreadpy as nfl


DB_PATH = Path("/tmp/nfl_prediction_lab_challenge.db")

STAT_LABELS = {
    "passing_yards": "Passing yards",
    "rushing_yards": "Rushing yards",
    "receiving_yards": "Receiving yards",
    "receptions": "Receptions",
    "passing_tds": "Passing TDs",
    "rushing_tds": "Rushing TDs",
    "receiving_tds": "Receiving TDs",
}

POSITION_STATS = {
    "QB": ["passing_yards", "passing_tds"],
    "RB": ["rushing_yards", "receiving_yards", "receptions", "rushing_tds", "receiving_tds"],
    "WR": ["receiving_yards", "receptions", "receiving_tds"],
    "TE": ["receiving_yards", "receptions", "receiving_tds"],
}


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS winner_picks (
            email TEXT NOT NULL,
            display_name TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            pick_team TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (email, game_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stat_predictions (
            email TEXT NOT NULL,
            display_name TEXT NOT NULL,
            season INTEGER NOT NULL,
            week INTEGER NOT NULL,
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            player_name TEXT NOT NULL,
            stat_key TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (email, game_id, player_id, stat_key)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS preferences (
            email TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            favorite_team TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def kickoff_datetime_et(game) -> datetime | None:
    gameday = pd.to_datetime(game.get("gameday"), errors="coerce")
    if pd.isna(gameday):
        return None

    raw_time = str(game.get("gametime") or "13:00")
    try:
        hour, minute = [int(x) for x in raw_time.split(":")[:2]]
    except Exception:
        hour, minute = 13, 0

    return datetime(
        gameday.year,
        gameday.month,
        gameday.day,
        hour,
        minute,
        tzinfo=ZoneInfo("America/New_York"),
    )


def game_is_open(game) -> bool:
    kickoff = kickoff_datetime_et(game)
    if kickoff is None:
        return True
    return datetime.now(ZoneInfo("America/New_York")) < kickoff


def save_winner_pick(email, display_name, season, week, game_id, pick_team):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO winner_picks
            (email, display_name, season, week, game_id, pick_team, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email, game_id) DO UPDATE SET
                display_name=excluded.display_name,
                pick_team=excluded.pick_team,
                created_at=excluded.created_at
            """,
            (
                email,
                display_name,
                int(season),
                int(week),
                str(game_id),
                str(pick_team),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


def winner_pick_for_user(email, game_id):
    with _connect() as conn:
        row = conn.execute(
            "SELECT pick_team FROM winner_picks WHERE email=? AND game_id=?",
            (email, str(game_id)),
        ).fetchone()
    return row["pick_team"] if row else None


def save_stat_prediction(
    email,
    display_name,
    season,
    week,
    game_id,
    player_id,
    player_name,
    stat_key,
    predicted_value,
):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO stat_predictions
            (email, display_name, season, week, game_id, player_id, player_name,
             stat_key, predicted_value, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(email, game_id, player_id, stat_key) DO UPDATE SET
                display_name=excluded.display_name,
                player_name=excluded.player_name,
                predicted_value=excluded.predicted_value,
                created_at=excluded.created_at
            """,
            (
                email,
                display_name,
                int(season),
                int(week),
                str(game_id),
                str(player_id),
                str(player_name),
                str(stat_key),
                float(predicted_value),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()


def stat_predictions_for_user(email, season, week):
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT * FROM stat_predictions
            WHERE email=? AND season=? AND week=?
            ORDER BY created_at DESC
            """,
            (email, int(season), int(week)),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def set_favorite_team(email, display_name, team):
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO preferences (email, display_name, favorite_team, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                display_name=excluded.display_name,
                favorite_team=excluded.favorite_team,
                updated_at=excluded.updated_at
            """,
            (email, display_name, team, datetime.utcnow().isoformat()),
        )
        conn.commit()


def favorite_team(email):
    with _connect() as conn:
        row = conn.execute(
            "SELECT favorite_team FROM preferences WHERE email=?",
            (email,),
        ).fetchone()
    return row["favorite_team"] if row else None


def load_skill_roster(season: int) -> pd.DataFrame:
    try:
        frame = nfl.load_rosters([int(season)])
        roster = frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)
    except Exception:
        return pd.DataFrame()

    if roster.empty:
        return roster

    for column in ["team", "full_name", "position", "gsis_id", "status"]:
        if column not in roster.columns:
            roster[column] = ""

    aliases = {
        "GNB": "GB", "KAN": "KC", "LAR": "LA", "NWE": "NE",
        "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV", "WSH": "WAS",
    }
    roster["team"] = roster["team"].astype(str).str.upper().map(
        lambda value: aliases.get(value, value)
    )
    roster["position"] = roster["position"].astype(str).str.upper()
    roster = roster[roster["position"].isin(POSITION_STATS)].copy()

    status = roster["status"].fillna("").astype(str).str.lower()
    practice = status.str.contains("practice")
    roster = roster[~practice].copy()
    roster = roster[roster["full_name"].fillna("").astype(str).str.strip().ne("")]
    return roster.drop_duplicates(["team", "gsis_id", "full_name"])


def load_weekly_player_stats(season: int) -> pd.DataFrame:
    try:
        frame = nfl.load_player_stats([int(season)], summary_level="week")
        stats = frame.to_pandas() if hasattr(frame, "to_pandas") else pd.DataFrame(frame)
    except Exception:
        return pd.DataFrame()

    if stats.empty:
        return stats

    player_id_col = "player_id" if "player_id" in stats.columns else "gsis_id"
    if player_id_col not in stats.columns:
        return pd.DataFrame()

    stats["player_id_norm"] = stats[player_id_col].astype(str)
    stats["week"] = pd.to_numeric(stats.get("week"), errors="coerce")
    return stats


def _actual_stat(stats, player_id, week, stat_key):
    if stats.empty or stat_key not in stats.columns:
        return None

    rows = stats[
        (stats["player_id_norm"] == str(player_id))
        & (pd.to_numeric(stats["week"], errors="coerce") == int(week))
    ]
    if rows.empty:
        return None

    value = pd.to_numeric(rows[stat_key], errors="coerce").dropna()
    if value.empty:
        return None
    return float(value.sum())


def _winner_for_game(row):
    home = pd.to_numeric(row.get("home_score"), errors="coerce")
    away = pd.to_numeric(row.get("away_score"), errors="coerce")
    if pd.isna(home) or pd.isna(away) or home == away:
        return None
    return str(row.get("home_team")) if home > away else str(row.get("away_team"))


def _stat_points(stat_key, predicted, actual):
    diff = abs(float(predicted) - float(actual))

    if stat_key == "passing_yards":
        if diff <= 25:
            return 10
        if diff <= 50:
            return 7
        if diff <= 100:
            return 3
        return 0

    if stat_key in {"rushing_yards", "receiving_yards"}:
        if diff <= 10:
            return 10
        if diff <= 25:
            return 7
        if diff <= 50:
            return 3
        return 0

    if stat_key == "receptions":
        if diff <= 1:
            return 10
        if diff <= 2:
            return 7
        if diff <= 3:
            return 3
        return 0

    if stat_key in {"passing_tds", "rushing_tds", "receiving_tds"}:
        if diff == 0:
            return 10
        if diff <= 1:
            return 5
        return 0

    return 0


def user_score(email, schedules, player_stats):
    winner_points = 0
    stat_points = 0
    settled_winners = 0
    settled_stats = 0

    schedule_map = {
        str(row.get("game_id")): row
        for _, row in schedules.iterrows()
        if pd.notna(row.get("game_id"))
    }

    with _connect() as conn:
        winner_rows = conn.execute(
            "SELECT * FROM winner_picks WHERE email=?",
            (email,),
        ).fetchall()
        stat_rows = conn.execute(
            "SELECT * FROM stat_predictions WHERE email=?",
            (email,),
        ).fetchall()

    for pick in winner_rows:
        game = schedule_map.get(str(pick["game_id"]))
        if game is None:
            continue
        winner = _winner_for_game(game)
        if winner is None:
            continue
        settled_winners += 1
        if str(pick["pick_team"]) == winner:
            winner_points += 10

    for pick in stat_rows:
        actual = _actual_stat(
            player_stats,
            pick["player_id"],
            pick["week"],
            pick["stat_key"],
        )
        if actual is None:
            continue
        settled_stats += 1
        stat_points += _stat_points(
            pick["stat_key"],
            pick["predicted_value"],
            actual,
        )

    return {
        "total": int(winner_points + stat_points),
        "winner_points": int(winner_points),
        "stat_points": int(stat_points),
        "settled_winners": int(settled_winners),
        "settled_stats": int(settled_stats),
    }


def leaderboard(schedules, player_stats):
    with _connect() as conn:
        users = conn.execute(
            """
            SELECT email, MAX(display_name) AS display_name
            FROM (
                SELECT email, display_name FROM winner_picks
                UNION ALL
                SELECT email, display_name FROM stat_predictions
                UNION ALL
                SELECT email, display_name FROM preferences
            )
            GROUP BY email
            """
        ).fetchall()

    rows = []
    for row in users:
        score = user_score(row["email"], schedules, player_stats)
        rows.append(
            {
                "Player": row["display_name"] or row["email"].split("@")[0],
                "Points": score["total"],
                "Winner picks": score["winner_points"],
                "Stat challenges": score["stat_points"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["Points", "Player"],
        ascending=[False, True],
    ).reset_index(drop=True) if rows else pd.DataFrame()
