from __future__ import annotations

import json
import re
from typing import Iterable

import numpy as np
import pandas as pd
import nflreadpy as nfl

from verified_transactions import VERIFIED_PLAYER_TRADES


POSITION_WEIGHTS = {
    "QB": 3.0,
    "LT": 1.8, "RT": 1.7, "T": 1.7, "OT": 1.7,
    "G": 1.4, "OG": 1.4, "C": 1.4,
    "WR": 1.5, "TE": 1.25, "RB": 1.0, "FB": 0.7,
    "EDGE": 1.7, "DE": 1.6, "OLB": 1.45, "DT": 1.35, "NT": 1.25,
    "LB": 1.25, "ILB": 1.25,
    "CB": 1.5, "S": 1.3, "FS": 1.3, "SS": 1.3, "DB": 1.25,
    "K": 0.55, "P": 0.45, "LS": 0.25,
}

# This multiplier is intentionally much flatter than POSITION_WEIGHTS.
# It is only used for the player-ranking UI so a reserve QB does not
# automatically outrank elite starters at other positions.
PROJECTED_POSITION_MULTIPLIER = {
    "QB": 1.12,
    "RB": 0.96,
    "WR": 1.04,
    "TE": 1.00,
    "OL": 1.07,
    "DL": 1.10,
    "LB": 1.00,
    "DB": 1.05,
    "ST": 0.75,
    "Other": 0.95,
}

TEAM_ALIASES = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV",
    "OAK": "LV", "SDG": "LAC", "STL": "LA", "WSH": "WAS",
}

OFFSEASON_FEATURES = [
    "roster_continuity",
    "addition_weight",
    "departure_weight",
    "draft_impact",
    "qb_returning",
    "trade_additions",
    "trade_departures",
]

OFFSEASON_DISPLAY_NAMES = {
    "roster_continuity": "Roster continuity",
    "addition_weight": "Veteran additions",
    "departure_weight": "Veteran departures",
    "draft_impact": "Draft class impact",
    "qb_returning": "Primary QB continuity",
    "trade_additions": "Trade additions",
    "trade_departures": "Trade departures",
}

OFFSEASON_DETAIL_COLUMNS = [
    "added_players",
    "departed_players",
    "drafted_players",
    "trade_added_players",
    "trade_departed_players",
    "added_player_details",
    "departed_player_details",
]


def _to_pandas(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _clean_name(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none"}:
        return ""
    return value


def _name_match_key(value) -> str:
    text = _clean_name(value).lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    parts = [p for p in text.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


def _norm_team(value):
    if pd.isna(value):
        return value
    value = str(value).upper().strip()
    return TEAM_ALIASES.get(value, value)


def _position_weight(position):
    if pd.isna(position):
        return 1.0
    return POSITION_WEIGHTS.get(str(position).upper(), 1.0)


def _position_group(position):
    pos = _clean_name(position).upper()
    if pos == "QB":
        return "QB"
    if pos in {"RB", "HB", "FB"}:
        return "RB"
    if pos == "WR":
        return "WR"
    if pos == "TE":
        return "TE"
    if pos in {"LT", "RT", "T", "OT", "G", "OG", "C", "OL"}:
        return "OL"
    if pos in {"EDGE", "DE", "DT", "NT", "DL"}:
        return "DL"
    if pos in {"LB", "ILB", "OLB", "MLB"}:
        return "LB"
    if pos in {"CB", "S", "FS", "SS", "DB"}:
        return "DB"
    if pos in {"K", "P", "LS"}:
        return "ST"
    return "Other"


def _player_key(row):
    gsis = row.get("gsis_id")
    if pd.notna(gsis) and str(gsis).strip():
        return f"gsis:{gsis}"
    name = str(row.get("full_name", "")).strip().lower()
    return f"name:{name}"


def _clean_roster(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    r = frame.copy()
    if r.empty:
        return r

    r["season"] = season
    r["team"] = r["team"].map(_norm_team)

    for column in ["full_name", "position", "status", "gsis_id", "pfr_id"]:
        if column not in r.columns:
            r[column] = ""

    if "years_exp" not in r.columns:
        r["years_exp"] = np.nan

    r["player_key"] = r.apply(_player_key, axis=1)
    r["position_weight"] = r["position"].map(_position_weight)
    return r.drop_duplicates(["season", "team", "player_key"])


def _verified_trade_frame(seasons: Iterable[int]) -> pd.DataFrame:
    season_set = {int(season) for season in seasons}
    rows = []
    for trade in VERIFIED_PLAYER_TRADES:
        season = int(trade.get("season", 0))
        if season not in season_set:
            continue
        rows.append(
            {
                "season": season,
                "gave": _norm_team(trade.get("from_team", "")),
                "received": _norm_team(trade.get("to_team", "")),
                "player_name": _clean_name(trade.get("player_name", "")),
                "position": _clean_name(trade.get("position", "")),
                "verified_supplement": True,
            }
        )
    return pd.DataFrame(rows)


def _merge_verified_trades(
    trades: pd.DataFrame,
    seasons: Iterable[int],
) -> pd.DataFrame:
    verified = _verified_trade_frame(seasons)
    if verified.empty:
        return trades.copy()

    base = trades.copy() if not trades.empty else pd.DataFrame()
    merged = pd.concat([base, verified], ignore_index=True, sort=False)

    for column in ["season", "gave", "received"]:
        if column not in merged.columns:
            merged[column] = ""

    name_candidates = [
        c
        for c in ["pfr_name", "player_name", "full_name", "name"]
        if c in merged.columns
    ]
    merged["_verified_name_key"] = ""
    for column in name_candidates:
        candidate = merged[column].map(_name_match_key)
        empty = merged["_verified_name_key"].eq("")
        merged.loc[empty, "_verified_name_key"] = candidate.loc[empty]

    merged["_verified_gave"] = merged["gave"].map(_norm_team)
    merged["_verified_received"] = merged["received"].map(_norm_team)
    merged["_verified_season"] = pd.to_numeric(
        merged["season"],
        errors="coerce",
    )

    merged = merged.drop_duplicates(
        [
            "_verified_season",
            "_verified_gave",
            "_verified_received",
            "_verified_name_key",
        ],
        keep="last",
    )

    return merged.drop(
        columns=[
            "_verified_name_key",
            "_verified_gave",
            "_verified_received",
            "_verified_season",
        ],
        errors="ignore",
    ).reset_index(drop=True)


def _apply_verified_trades_to_rosters(
    rosters: pd.DataFrame,
    seasons: Iterable[int],
) -> pd.DataFrame:
    if rosters.empty:
        return rosters

    r = rosters.copy()
    r["_name_match_key"] = r["full_name"].map(_name_match_key)
    season_set = {int(season) for season in seasons}

    additions = []
    for trade in VERIFIED_PLAYER_TRADES:
        season = int(trade.get("season", 0))
        if season not in season_set:
            continue

        from_team = _norm_team(trade.get("from_team", ""))
        to_team = _norm_team(trade.get("to_team", ""))
        player_name = _clean_name(trade.get("player_name", ""))
        position = _clean_name(trade.get("position", ""))
        name_key = _name_match_key(player_name)
        if not name_key or not from_team or not to_team:
            continue

        current_mask = (
            (pd.to_numeric(r["season"], errors="coerce") == season)
            & (r["_name_match_key"] == name_key)
        )
        current_matches = r[current_mask].copy()

        source = current_matches[current_matches["team"] == from_team]
        if source.empty:
            prior = r[
                (pd.to_numeric(r["season"], errors="coerce") == season - 1)
                & (r["_name_match_key"] == name_key)
            ]
            source = prior
        if source.empty:
            source = current_matches

        # Remove stale pre-trade placement from the old team.
        r = r[
            ~(
                (pd.to_numeric(r["season"], errors="coerce") == season)
                & (r["team"] == from_team)
                & (r["_name_match_key"] == name_key)
            )
        ].copy()

        already_on_new_team = (
            (pd.to_numeric(r["season"], errors="coerce") == season)
            & (r["team"] == to_team)
            & (r["_name_match_key"] == name_key)
        ).any()
        if already_on_new_team:
            continue

        if not source.empty:
            row = source.iloc[0].to_dict()
        else:
            row = {column: np.nan for column in r.columns}
            row.update(
                {
                    "full_name": player_name,
                    "position": position,
                    "status": "Active",
                    "gsis_id": "",
                    "pfr_id": "",
                    "years_exp": np.nan,
                }
            )

        row["season"] = season
        row["team"] = to_team
        row["full_name"] = _clean_name(row.get("full_name")) or player_name
        row["position"] = _clean_name(row.get("position")) or position
        row["status"] = _clean_name(row.get("status")) or "Active"
        row["_name_match_key"] = name_key

        row_series = pd.Series(row)
        row["player_key"] = _player_key(row_series)
        row["position_weight"] = _position_weight(row.get("position"))
        additions.append(row)

    if additions:
        r = pd.concat([r, pd.DataFrame(additions)], ignore_index=True, sort=False)

    return (
        r.drop(columns=["_name_match_key"], errors="ignore")
        .drop_duplicates(["season", "team", "player_key"], keep="last")
        .reset_index(drop=True)
    )


def _clean_depth_charts(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    d = frame.copy()
    if d.empty:
        return d

    d["season"] = season
    if "team" in d.columns:
        d["team"] = d["team"].map(_norm_team)

    if "dt" in d.columns:
        d["dt"] = pd.to_datetime(d["dt"], errors="coerce", utc=True)

    for column in ["player_name", "gsis_id", "pos_abb"]:
        if column not in d.columns:
            d[column] = ""

    if "pos_rank" not in d.columns:
        # Legacy depth-chart files use different naming. The latest-season
        # 2025+ schema supplies pos_rank; older rows simply fall back to
        # recent snaps for UI ranking.
        d["pos_rank"] = np.nan

    d["pos_rank"] = pd.to_numeric(d["pos_rank"], errors="coerce")
    return d


def load_offseason_source_data(seasons: Iterable[int]):
    seasons = sorted(set(int(s) for s in seasons))
    roster_years = sorted(set([min(seasons) - 1] + seasons))
    latest_season = max(seasons)

    roster_frames = []
    unavailable_rosters = []
    for year in roster_years:
        try:
            roster_frames.append(
                _clean_roster(_to_pandas(nfl.load_rosters([year])), year)
            )
        except Exception:
            unavailable_rosters.append(year)

    rosters = (
        pd.concat(roster_frames, ignore_index=True, sort=False)
        if roster_frames
        else pd.DataFrame()
    )
    rosters = _apply_verified_trades_to_rosters(rosters, seasons)

    try:
        drafts = _to_pandas(nfl.load_draft_picks(seasons))
    except Exception:
        drafts = pd.DataFrame()

    try:
        trades = _to_pandas(nfl.load_trades())
    except Exception:
        trades = pd.DataFrame()

    trades = _merge_verified_trades(trades, seasons)

    # Three completed seasons drive the current player-ranking UI, while
    # every season's prior year is retained for historical QB-continuity
    # features used by the model.
    history_years = list(range(max(latest_season - 3, 1999), latest_season))
    stats_years = sorted(set(history_years + [season - 1 for season in seasons]))

    try:
        player_stats = _to_pandas(
            nfl.load_player_stats(stats_years, summary_level="reg")
        )
    except Exception:
        player_stats = pd.DataFrame()

    try:
        snap_counts = _to_pandas(nfl.load_snap_counts(stats_years))
    except Exception:
        snap_counts = pd.DataFrame()

    try:
        depth_charts = _clean_depth_charts(
            _to_pandas(nfl.load_depth_charts([latest_season])),
            latest_season,
        )
    except Exception:
        depth_charts = pd.DataFrame()

    return (
        rosters,
        drafts,
        trades,
        player_stats,
        snap_counts,
        depth_charts,
        unavailable_rosters,
    )


def _primary_passers(player_stats: pd.DataFrame) -> dict[tuple[int, str], str]:
    if (
        player_stats.empty
        or "team" not in player_stats.columns
        or "player_id" not in player_stats.columns
    ):
        return {}

    p = player_stats.copy()
    p["team"] = p["team"].map(_norm_team)
    attempts_col = next(
        (c for c in ["attempts", "passing_attempts"] if c in p.columns),
        None,
    )
    if attempts_col is None:
        p["_attempts"] = 0.0
        attempts_col = "_attempts"

    p[attempts_col] = pd.to_numeric(p[attempts_col], errors="coerce").fillna(0)
    p = p.sort_values(attempts_col, ascending=False).drop_duplicates(
        ["season", "team"]
    )

    return {
        (int(row.season), str(row.team)): str(row.player_id)
        for row in p.itertuples()
        if pd.notna(row.player_id)
    }


def _draft_rows_for_team(
    drafts: pd.DataFrame,
    season: int,
    team: str,
) -> pd.DataFrame:
    if drafts.empty or "season" not in drafts.columns or "team" not in drafts.columns:
        return pd.DataFrame()

    d = drafts.copy()
    d["team"] = d["team"].map(_norm_team)
    return d[
        (pd.to_numeric(d["season"], errors="coerce") == season)
        & (d["team"] == team)
    ].copy()


def _draft_impact_for_team(
    drafts: pd.DataFrame,
    season: int,
    team: str,
) -> float:
    d = _draft_rows_for_team(drafts, season, team)
    if d.empty:
        return 0.0

    total = 0.0
    for row in d.itertuples():
        pick = pd.to_numeric(getattr(row, "pick", np.nan), errors="coerce")
        position = getattr(row, "position", "")
        if pd.isna(pick):
            continue
        pick_value = float(np.exp(-(float(pick) - 1.0) / 85.0))
        total += pick_value * _position_weight(position)
    return float(total)


def _draft_score_for_player(
    drafts: pd.DataFrame,
    season: int,
    team: str,
    player_id,
    player_name: str,
) -> float:
    d = _draft_rows_for_team(drafts, season, team)
    if d.empty:
        return 0.0

    match = pd.DataFrame()
    if "gsis_id" in d.columns and pd.notna(player_id) and str(player_id).strip():
        match = d[d["gsis_id"].astype(str) == str(player_id)].copy()

    if match.empty:
        name_col = next(
            (
                c
                for c in ["pfr_player_name", "player_name", "full_name", "name"]
                if c in d.columns
            ),
            None,
        )
        if name_col:
            target = player_name.strip().lower()
            match = d[
                d[name_col].fillna("").astype(str).str.strip().str.lower() == target
            ].copy()

    if match.empty or "pick" not in match.columns:
        return 0.0

    pick = pd.to_numeric(match["pick"], errors="coerce").dropna()
    if pick.empty:
        return 0.0

    # 1st overall ~= 1.00, pick 32 ~= .73, pick 100 ~= .37.
    return float(np.exp(-(float(pick.min()) - 1.0) / 100.0))


def _stat_total(rows: pd.DataFrame, names) -> float:
    if rows.empty:
        return 0.0
    for name in names:
        if name in rows.columns:
            return float(
                pd.to_numeric(rows[name], errors="coerce").fillna(0).sum()
            )
    return 0.0


def _season_production_score(rows: pd.DataFrame, position: str) -> float:
    if rows.empty:
        return 0.0

    group = _position_group(position)

    if group == "QB":
        attempts = _stat_total(rows, ["attempts", "passing_attempts"])
        yards = _stat_total(rows, ["passing_yards"])
        touchdowns = _stat_total(rows, ["passing_tds"])
        epa = _stat_total(rows, ["passing_epa"])
        volume = min(attempts / 550.0, 1.0)
        yard_score = min(yards / 4500.0, 1.0)
        td_score = min(touchdowns / 35.0, 1.0)
        epa_score = float(np.clip((epa + 20.0) / 120.0, 0.0, 1.0))
        return float(
            0.25 * volume
            + 0.25 * yard_score
            + 0.20 * td_score
            + 0.30 * epa_score
        )

    if group == "RB":
        rush_yards = _stat_total(rows, ["rushing_yards"])
        receiving_yards = _stat_total(rows, ["receiving_yards"])
        touchdowns = (
            _stat_total(rows, ["rushing_tds"])
            + _stat_total(rows, ["receiving_tds"])
        )
        return float(
            np.clip(
                0.55 * min(rush_yards / 1200.0, 1.0)
                + 0.25 * min(receiving_yards / 600.0, 1.0)
                + 0.20 * min(touchdowns / 12.0, 1.0),
                0.0,
                1.0,
            )
        )

    if group in {"WR", "TE"}:
        yards_target = 1300.0 if group == "WR" else 950.0
        target_target = 150.0 if group == "WR" else 120.0
        td_target = 12.0 if group == "WR" else 10.0
        receiving_yards = _stat_total(rows, ["receiving_yards"])
        targets = _stat_total(rows, ["targets"])
        touchdowns = _stat_total(rows, ["receiving_tds"])
        return float(
            np.clip(
                0.55 * min(receiving_yards / yards_target, 1.0)
                + 0.25 * min(targets / target_target, 1.0)
                + 0.20 * min(touchdowns / td_target, 1.0),
                0.0,
                1.0,
            )
        )

    if group in {"DL", "LB", "DB"}:
        tackles = _stat_total(
            rows,
            ["def_tackles", "def_tackles_solo", "def_tackles_combined"],
        )
        sacks = _stat_total(rows, ["def_sacks", "sacks"])
        tfl = _stat_total(rows, ["def_tackles_for_loss"])
        qb_hits = _stat_total(rows, ["def_qb_hits"])
        interceptions = _stat_total(
            rows,
            ["def_interceptions", "interceptions"],
        )
        passes_defended = _stat_total(rows, ["def_pass_defended"])
        forced_fumbles = _stat_total(rows, ["def_fumbles_forced"])

        if group == "DL":
            score = (
                0.15 * min(tackles / 80.0, 1.0)
                + 0.35 * min(sacks / 15.0, 1.0)
                + 0.20 * min(tfl / 20.0, 1.0)
                + 0.20 * min(qb_hits / 25.0, 1.0)
                + 0.10 * min(forced_fumbles / 6.0, 1.0)
            )
        elif group == "LB":
            score = (
                0.35 * min(tackles / 130.0, 1.0)
                + 0.20 * min(sacks / 10.0, 1.0)
                + 0.15 * min(tfl / 18.0, 1.0)
                + 0.15 * min(interceptions / 5.0, 1.0)
                + 0.15 * min(forced_fumbles / 6.0, 1.0)
            )
        else:
            score = (
                0.25 * min(tackles / 100.0, 1.0)
                + 0.30 * min(interceptions / 6.0, 1.0)
                + 0.30 * min(passes_defended / 18.0, 1.0)
                + 0.15 * min(forced_fumbles / 5.0, 1.0)
            )
        return float(np.clip(score, 0.0, 1.0))

    # OL and special teams are evaluated mainly through snaps/role because
    # standard player-stat tables do not measure their value well.
    return 0.0


def _recent_production_score(
    player_stats: pd.DataFrame,
    player_id,
    position: str,
    prior_season: int,
) -> float:
    if (
        player_stats.empty
        or pd.isna(player_id)
        or not str(player_id).strip()
        or "player_id" not in player_stats.columns
    ):
        return 0.0

    rows = player_stats[player_stats["player_id"].astype(str) == str(player_id)].copy()
    if rows.empty:
        return 0.0

    season_scores = {}
    for season in range(prior_season - 2, prior_season + 1):
        season_rows = rows[
            pd.to_numeric(rows.get("season"), errors="coerce") == season
        ]
        season_scores[season] = _season_production_score(
            season_rows,
            position,
        )

    weights = {
        prior_season: 0.60,
        prior_season - 1: 0.25,
        prior_season - 2: 0.15,
    }
    weighted = sum(
        season_scores.get(season, 0.0) * weight
        for season, weight in weights.items()
    )
    peak = max(season_scores.values(), default=0.0)

    # Peak component helps recognize established veterans returning from an
    # injury/absence, while recency still carries most of the weight.
    return float(np.clip(0.75 * weighted + 0.25 * peak, 0.0, 1.0))


def _normalize_pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    if not values.empty and values.max() > 1.5:
        values = values / 100.0
    return values.clip(lower=0.0, upper=1.0)


def _recent_snap_score(
    snap_counts: pd.DataFrame,
    pfr_id,
    player_name: str,
    prior_season: int,
) -> float:
    if snap_counts.empty:
        return 0.0

    rows = pd.DataFrame()
    if (
        "pfr_player_id" in snap_counts.columns
        and pd.notna(pfr_id)
        and str(pfr_id).strip()
    ):
        rows = snap_counts[
            snap_counts["pfr_player_id"].astype(str) == str(pfr_id)
        ].copy()

    if rows.empty and "player" in snap_counts.columns:
        target = player_name.strip().lower()
        rows = snap_counts[
            snap_counts["player"].fillna("").astype(str).str.strip().str.lower()
            == target
        ].copy()

    if rows.empty:
        return 0.0

    season_scores = {}
    for season in range(prior_season - 2, prior_season + 1):
        season_rows = rows[
            pd.to_numeric(rows.get("season"), errors="coerce") == season
        ].copy()
        if season_rows.empty:
            season_scores[season] = 0.0
            continue

        pct_columns = [
            c
            for c in ["offense_pct", "defense_pct"]
            if c in season_rows.columns
        ]
        if not pct_columns:
            season_scores[season] = 0.0
            continue

        pct_values = pd.concat(
            [_normalize_pct(season_rows[c]) for c in pct_columns],
            axis=1,
        )
        game_share = pct_values.max(axis=1)
        season_scores[season] = float(game_share.mean())

    weights = {
        prior_season: 0.60,
        prior_season - 1: 0.25,
        prior_season - 2: 0.15,
    }
    weighted = sum(
        season_scores.get(season, 0.0) * weight
        for season, weight in weights.items()
    )
    peak = max(season_scores.values(), default=0.0)
    return float(np.clip(0.75 * weighted + 0.25 * peak, 0.0, 1.0))


def _latest_depth_rows(
    depth_charts: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    if depth_charts.empty or "team" not in depth_charts.columns:
        return pd.DataFrame()

    rows = depth_charts[depth_charts["team"] == team].copy()
    if rows.empty:
        return rows

    if "dt" in rows.columns and rows["dt"].notna().any():
        latest_dt = rows["dt"].max()
        rows = rows[rows["dt"] == latest_dt].copy()

    return rows


def _depth_rank_for_player(
    depth_charts: pd.DataFrame,
    team: str,
    player_id,
    player_name: str,
) -> float | None:
    rows = _latest_depth_rows(depth_charts, team)
    if rows.empty:
        return None

    match = pd.DataFrame()
    if "gsis_id" in rows.columns and pd.notna(player_id) and str(player_id).strip():
        match = rows[rows["gsis_id"].astype(str) == str(player_id)].copy()

    if match.empty and "player_name" in rows.columns:
        target = player_name.strip().lower()
        match = rows[
            rows["player_name"].fillna("").astype(str).str.strip().str.lower()
            == target
        ].copy()

    if match.empty:
        return None

    rank = pd.to_numeric(match["pos_rank"], errors="coerce").dropna()
    if rank.empty:
        return None
    return float(rank.min())


def _roster_status_factor(status) -> tuple[float, str]:
    text = _clean_name(status).lower()

    if "practice" in text or text in {"pra", "ps"}:
        return 0.04, "Practice squad"
    if "injured reserve" in text or text in {"ir", "reserve/injured"}:
        return 0.18, "Injured reserve"
    if "suspend" in text:
        return 0.08, "Suspended"
    if "inactive" in text:
        return 0.15, "Inactive"
    return 1.0, "Active roster"


def _addition_role_factor(
    depth_charts: pd.DataFrame,
    team: str,
    player_id,
    player_name: str,
    position: str,
    roster_status,
) -> tuple[float, str]:
    status_factor, status_label = _roster_status_factor(roster_status)
    if status_factor < 0.25:
        return status_factor, status_label

    rank = _depth_rank_for_player(
        depth_charts,
        team,
        player_id,
        player_name,
    )
    group = _position_group(position)

    if rank is None:
        return 0.20 * status_factor, "Active / depth role not listed"

    rank_int = max(1, int(round(rank)))

    if group == "QB":
        factors = {1: 1.00, 2: 0.28, 3: 0.08}
        factor = factors.get(rank_int, 0.03)
        label = (
            "QB1 / Starter"
            if rank_int == 1
            else f"QB{rank_int} / Backup"
        )
    elif group == "RB":
        factors = {1: 0.95, 2: 0.68, 3: 0.45, 4: 0.28}
        factor = factors.get(rank_int, 0.18)
        label = f"RB depth rank {rank_int}"
    elif group == "WR":
        factors = {1: 0.95, 2: 0.90, 3: 0.82, 4: 0.58, 5: 0.38}
        factor = factors.get(rank_int, 0.25)
        label = f"WR depth rank {rank_int}"
    elif group == "TE":
        factors = {1: 0.95, 2: 0.68, 3: 0.42}
        factor = factors.get(rank_int, 0.25)
        label = f"TE depth rank {rank_int}"
    elif group == "OL":
        factors = {1: 0.98, 2: 0.45, 3: 0.22}
        factor = factors.get(rank_int, 0.15)
        label = "Starter" if rank_int == 1 else f"OL depth rank {rank_int}"
    elif group in {"DL", "LB", "DB"}:
        factors = {1: 0.98, 2: 0.78, 3: 0.55, 4: 0.35}
        factor = factors.get(rank_int, 0.25)
        label = "Starter / top unit" if rank_int == 1 else f"Depth rank {rank_int}"
    elif group == "ST":
        factors = {1: 0.90, 2: 0.35}
        factor = factors.get(rank_int, 0.15)
        label = "Primary specialist" if rank_int == 1 else f"ST depth rank {rank_int}"
    else:
        factor = 0.85 if rank_int == 1 else max(0.20, 0.55 / rank_int)
        label = "Starter" if rank_int == 1 else f"Depth rank {rank_int}"

    return float(factor * status_factor), label


def _departure_role_factor(
    snap_score: float,
    position: str,
) -> tuple[float, str]:
    group = _position_group(position)

    if group == "QB":
        if snap_score >= 0.65:
            return min(1.0, 0.35 + 0.75 * snap_score), "Former starting QB"
        if snap_score >= 0.20:
            return 0.30, "Former QB2 / spot starter"
        return 0.10, "Former reserve QB"

    if snap_score >= 0.70:
        return min(1.0, 0.35 + 0.75 * snap_score), "Former starter"
    if snap_score >= 0.35:
        return 0.55, "Former rotation player"
    if snap_score >= 0.10:
        return 0.30, "Former reserve / role player"
    return 0.15, "Limited prior role"


def _projected_impact_detail(
    row,
    team: str,
    season: int,
    direction: str,
    drafts: pd.DataFrame,
    player_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
) -> dict | None:
    name = _clean_name(getattr(row, "full_name", ""))
    if not name:
        return None

    position = _clean_name(getattr(row, "position", "")).upper() or "—"
    group = _position_group(position)
    player_id = getattr(row, "gsis_id", None)
    pfr_id = getattr(row, "pfr_id", None)
    prior_season = season - 1

    production = _recent_production_score(
        player_stats,
        player_id,
        position,
        prior_season,
    )
    snap_score = _recent_snap_score(
        snap_counts,
        pfr_id,
        name,
        prior_season,
    )
    draft_score = _draft_score_for_player(
        drafts,
        season,
        team,
        player_id,
        name,
    )

    if direction == "addition":
        role_factor, role_label = _addition_role_factor(
            depth_charts,
            team,
            player_id,
            name,
            position,
            getattr(row, "status", ""),
        )

        # Current role drives the projection. Production and snap history
        # describe proven NFL ability; draft capital helps rookies who do not
        # have NFL stats yet.
        talent = (
            0.25
            + 0.45 * production
            + 0.20 * snap_score
            + 0.25 * draft_score
        )
    else:
        role_factor, role_label = _departure_role_factor(
            snap_score,
            position,
        )
        talent = (
            0.25
            + 0.50 * production
            + 0.25 * snap_score
        )

    position_multiplier = PROJECTED_POSITION_MULTIPLIER.get(group, 0.95)
    impact_score = float(
        np.clip(
            100.0 * role_factor * position_multiplier * talent,
            0.0,
            100.0,
        )
    )

    return {
        "name": name,
        "position": position,
        "position_group": group,
        "impact_score": round(impact_score, 1),
        "projected_role": role_label,
        "role_factor": round(float(role_factor), 3),
        "recent_production": round(float(production), 3),
        "recent_snap_share": round(float(snap_score), 3),
        "draft_score": round(float(draft_score), 3),
    }


def _roster_player_details(
    roster: pd.DataFrame,
    keys: set[str],
    team: str,
    season: int,
    direction: str,
    drafts: pd.DataFrame,
    player_stats: pd.DataFrame,
    snap_counts: pd.DataFrame,
    depth_charts: pd.DataFrame,
) -> str:
    if roster.empty or not keys:
        return "[]"

    selected = roster[roster["player_key"].isin(keys)].copy()
    details = []

    for row in selected.itertuples():
        detail = _projected_impact_detail(
            row,
            team,
            season,
            direction,
            drafts,
            player_stats,
            snap_counts,
            depth_charts,
        )
        if detail:
            details.append(detail)

    details.sort(
        key=lambda item: (-item["impact_score"], item["name"])
    )
    return json.dumps(details)


def _format_name(name, position="", pick=None) -> str:
    name = _clean_name(name)
    if not name:
        return ""

    details = []
    position = _clean_name(position)
    if position:
        details.append(position)

    if pick is not None:
        pick_value = pd.to_numeric(pick, errors="coerce")
        if pd.notna(pick_value):
            details.append(f"pick {int(pick_value)}")

    return f"{name} ({', '.join(details)})" if details else name


def _roster_player_names(
    roster: pd.DataFrame,
    keys: set[str],
) -> str:
    if roster.empty or not keys:
        return ""

    selected = roster[roster["player_key"].isin(keys)].copy()
    names = []

    for row in selected.itertuples():
        label = _format_name(
            getattr(row, "full_name", ""),
            getattr(row, "position", ""),
        )
        if label:
            names.append(label)

    return " | ".join(sorted(set(names)))


def _draft_player_names(
    drafts: pd.DataFrame,
    season: int,
    team: str,
) -> str:
    d = _draft_rows_for_team(drafts, season, team)
    if d.empty:
        return ""

    name_col = next(
        (
            c
            for c in ["pfr_player_name", "player_name", "full_name", "name"]
            if c in d.columns
        ),
        None,
    )
    if not name_col:
        return ""

    names = []
    for row in d.itertuples():
        label = _format_name(
            getattr(row, name_col, ""),
            getattr(row, "position", ""),
            getattr(row, "pick", None),
        )
        if label:
            names.append(label)

    return " | ".join(names)


def _trade_details(
    trades: pd.DataFrame,
    season: int,
    team: str,
) -> tuple[float, float, str, str]:
    if trades.empty or "season" not in trades.columns:
        return 0.0, 0.0, "", ""

    t = trades.copy()
    t = t[pd.to_numeric(t["season"], errors="coerce") == season]
    if t.empty:
        return 0.0, 0.0, "", ""

    if "gave" in t.columns:
        t["gave"] = t["gave"].map(_norm_team)
    if "received" in t.columns:
        t["received"] = t["received"].map(_norm_team)

    name_candidates = [
        c
        for c in ["pfr_name", "player_name", "full_name", "name"]
        if c in t.columns
    ]
    name_col = None
    if name_candidates:
        t["_trade_player_name"] = ""
        for column in name_candidates:
            values = t[column].map(_clean_name)
            empty = t["_trade_player_name"].eq("")
            t.loc[empty, "_trade_player_name"] = values.loc[empty]
        name_col = "_trade_player_name"
        player_rows = t[t[name_col].ne("")].copy()
    elif "pfr_id" in t.columns:
        player_rows = t[t["pfr_id"].notna()].copy()
    else:
        player_rows = t.copy()

    additions_mask = (
        player_rows.get("received", pd.Series("", index=player_rows.index))
        == team
    )
    departures_mask = (
        player_rows.get("gave", pd.Series("", index=player_rows.index))
        == team
    )

    additions = float(additions_mask.sum())
    departures = float(departures_mask.sum())

    addition_names = ""
    departure_names = ""
    if name_col:
        addition_names = " | ".join(
            sorted(
                {
                    _clean_name(v)
                    for v in player_rows.loc[additions_mask, name_col]
                    if _clean_name(v)
                }
            )
        )
        departure_names = " | ".join(
            sorted(
                {
                    _clean_name(v)
                    for v in player_rows.loc[departures_mask, name_col]
                    if _clean_name(v)
                }
            )
        )

    return additions, departures, addition_names, departure_names


def build_offseason_features(
    seasons: Iterable[int],
) -> pd.DataFrame:
    seasons = sorted(set(int(s) for s in seasons))

    (
        rosters,
        drafts,
        trades,
        player_stats,
        snap_counts,
        depth_charts,
        unavailable,
    ) = load_offseason_source_data(seasons)

    primary_qbs = _primary_passers(player_stats)

    if rosters.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "team",
            ]
            + OFFSEASON_FEATURES
            + OFFSEASON_DETAIL_COLUMNS
        )

    teams = sorted(rosters["team"].dropna().astype(str).unique())
    rows = []
    latest_season = max(seasons)

    for season in seasons:
        prior_season = season - 1

        for team in teams:
            prior = rosters[
                (rosters["season"] == prior_season)
                & (rosters["team"] == team)
            ]
            current = rosters[
                (rosters["season"] == season)
                & (rosters["team"] == team)
            ]

            if prior.empty or current.empty:
                continue

            prior_map = prior.set_index("player_key")[
                "position_weight"
            ].to_dict()
            current_map = current.set_index("player_key")[
                "position_weight"
            ].to_dict()

            prior_keys = set(prior_map)
            current_keys = set(current_map)
            retained = prior_keys & current_keys
            additions = current_keys - prior_keys
            departures = prior_keys - current_keys

            prior_weight = max(sum(prior_map.values()), 1.0)
            current_weight = max(sum(current_map.values()), 1.0)

            continuity = (
                sum(prior_map[k] for k in retained) / prior_weight
            )
            addition_weight = (
                sum(current_map[k] for k in additions) / current_weight
            )
            departure_weight = (
                sum(prior_map[k] for k in departures) / prior_weight
            )

            primary_qb = primary_qbs.get((prior_season, team))
            current_gsis = set(
                current.get(
                    "gsis_id",
                    pd.Series(dtype=str),
                )
                .dropna()
                .astype(str)
            )
            qb_returning = (
                1.0
                if primary_qb and primary_qb in current_gsis
                else 0.0
            )

            (
                trade_additions,
                trade_departures,
                trade_added_players,
                trade_departed_players,
            ) = _trade_details(trades, season, team)

            # Depth charts are loaded for the latest season only because the
            # UI displays the current offseason. Historical model features
            # remain unchanged.
            season_depth = (
                depth_charts
                if season == latest_season
                else pd.DataFrame()
            )

            rows.append(
                {
                    "season": season,
                    "team": team,
                    "roster_continuity": float(continuity),
                    "addition_weight": float(addition_weight),
                    "departure_weight": float(departure_weight),
                    "draft_impact": _draft_impact_for_team(
                        drafts,
                        season,
                        team,
                    ),
                    "qb_returning": qb_returning,
                    "trade_additions": trade_additions,
                    "trade_departures": trade_departures,
                    "added_players": _roster_player_names(
                        current,
                        additions,
                    ),
                    "departed_players": _roster_player_names(
                        prior,
                        departures,
                    ),
                    "added_player_details": _roster_player_details(
                        current,
                        additions,
                        team,
                        season,
                        "addition",
                        drafts,
                        player_stats,
                        snap_counts,
                        season_depth,
                    ),
                    "departed_player_details": _roster_player_details(
                        prior,
                        departures,
                        team,
                        season,
                        "departure",
                        drafts,
                        player_stats,
                        snap_counts,
                        season_depth,
                    ),
                    "drafted_players": _draft_player_names(
                        drafts,
                        season,
                        team,
                    ),
                    "trade_added_players": trade_added_players,
                    "trade_departed_players": trade_departed_players,
                }
            )

    result = pd.DataFrame(rows)
    result.attrs["unavailable_roster_years"] = unavailable
    return result


def team_offseason_snapshot(
    offseason: pd.DataFrame,
    team: str,
    season: int,
) -> dict:
    if offseason.empty:
        return {f: 0.0 for f in OFFSEASON_FEATURES}

    row = offseason[
        (offseason["team"] == team)
        & (offseason["season"] == season)
    ]
    if row.empty:
        return {f: 0.0 for f in OFFSEASON_FEATURES}

    r = row.iloc[-1]
    return {
        f: float(r.get(f, 0.0))
        for f in OFFSEASON_FEATURES
    }
