from __future__ import annotations

from typing import Iterable
import json

import numpy as np
import pandas as pd
import nflreadpy as nfl


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

TEAM_ALIASES = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV",
    "OAK": "LV", "SDG": "LAC", "STL": "LA",
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


def _norm_team(value):
    if pd.isna(value):
        return value
    value = str(value).upper()
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


def _stat_total(rows: pd.DataFrame, names) -> float:
    if rows.empty:
        return 0.0
    for name in names:
        if name in rows.columns:
            return float(pd.to_numeric(rows[name], errors="coerce").fillna(0).sum())
    return 0.0


def _prior_usage_score(player_stats: pd.DataFrame, player_id, season: int, position: str) -> float:
    if player_stats.empty or pd.isna(player_id) or "player_id" not in player_stats.columns:
        return 0.0

    rows = player_stats[
        (pd.to_numeric(player_stats.get("season"), errors="coerce") == season)
        & (player_stats["player_id"].astype(str) == str(player_id))
    ]
    if rows.empty:
        return 0.0

    group = _position_group(position)
    if group == "QB":
        attempts = _stat_total(rows, ["passing_attempts"])
        return min(attempts / 350.0, 1.5)
    if group == "RB":
        carries = _stat_total(rows, ["carries", "rushing_attempts"])
        targets = _stat_total(rows, ["targets"])
        return min((carries / 180.0) + (targets / 90.0), 1.5)
    if group in {"WR", "TE"}:
        targets = _stat_total(rows, ["targets"])
        return min(targets / 100.0, 1.5)

    tackles = _stat_total(rows, ["tackles", "def_tackles_solo", "def_tackles_combined"])
    sacks = _stat_total(rows, ["sacks", "def_sacks"])
    interceptions = _stat_total(rows, ["interceptions", "def_interceptions"])
    defensive_usage = (tackles / 100.0) + (sacks / 12.0) + (interceptions / 6.0)
    return min(defensive_usage, 1.5)


def _roster_player_details(
    roster: pd.DataFrame,
    keys: set[str],
    player_stats: pd.DataFrame,
    prior_season: int,
) -> str:
    if roster.empty or not keys:
        return "[]"

    selected = roster[roster["player_key"].isin(keys)].copy()
    details = []
    for row in selected.itertuples():
        name = _clean_name(getattr(row, "full_name", ""))
        if not name:
            continue

        position = _clean_name(getattr(row, "position", "")).upper() or "—"
        player_id = getattr(row, "gsis_id", None)
        base_weight = float(_position_weight(position))
        usage = _prior_usage_score(player_stats, player_id, prior_season, position)
        impact_score = base_weight * (1.0 + 0.35 * usage)

        details.append({
            "name": name,
            "position": position,
            "position_group": _position_group(position),
            "impact_score": round(float(impact_score), 3),
            "position_weight": round(base_weight, 3),
            "prior_usage": round(float(usage), 3),
        })

    details.sort(key=lambda item: (-item["impact_score"], item["name"]))
    return json.dumps(details)


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
    if "full_name" not in r.columns:
        r["full_name"] = ""
    if "position" not in r.columns:
        r["position"] = ""
    r["player_key"] = r.apply(_player_key, axis=1)
    r["position_weight"] = r["position"].map(_position_weight)
    return r.drop_duplicates(["season", "team", "player_key"])


def load_offseason_source_data(seasons: Iterable[int]):
    seasons = sorted(set(int(s) for s in seasons))
    roster_years = sorted(set([min(seasons) - 1] + seasons))

    roster_frames = []
    unavailable_rosters = []
    for year in roster_years:
        try:
            roster_frames.append(_clean_roster(_to_pandas(nfl.load_rosters([year])), year))
        except Exception:
            unavailable_rosters.append(year)

    rosters = pd.concat(roster_frames, ignore_index=True, sort=False) if roster_frames else pd.DataFrame()

    try:
        drafts = _to_pandas(nfl.load_draft_picks(seasons))
    except Exception:
        drafts = pd.DataFrame()

    try:
        trades = _to_pandas(nfl.load_trades())
    except Exception:
        trades = pd.DataFrame()

    prior_years = sorted(set(s - 1 for s in seasons))
    try:
        player_stats = _to_pandas(nfl.load_player_stats(prior_years, summary_level="reg"))
    except Exception:
        player_stats = pd.DataFrame()

    return rosters, drafts, trades, player_stats, unavailable_rosters


def _primary_passers(player_stats: pd.DataFrame) -> dict[tuple[int, str], str]:
    if player_stats.empty or "team" not in player_stats.columns or "player_id" not in player_stats.columns:
        return {}

    p = player_stats.copy()
    p["team"] = p["team"].map(_norm_team)
    if "passing_attempts" not in p.columns:
        p["passing_attempts"] = 0
    p["passing_attempts"] = pd.to_numeric(p["passing_attempts"], errors="coerce").fillna(0)
    p = p.sort_values("passing_attempts", ascending=False).drop_duplicates(["season", "team"])
    return {
        (int(row.season), str(row.team)): str(row.player_id)
        for row in p.itertuples()
        if pd.notna(row.player_id)
    }


def _draft_rows_for_team(drafts: pd.DataFrame, season: int, team: str) -> pd.DataFrame:
    if drafts.empty or "season" not in drafts.columns or "team" not in drafts.columns:
        return pd.DataFrame()
    d = drafts.copy()
    d["team"] = d["team"].map(_norm_team)
    return d[
        (pd.to_numeric(d["season"], errors="coerce") == season)
        & (d["team"] == team)
    ].copy()


def _draft_impact_for_team(drafts: pd.DataFrame, season: int, team: str) -> float:
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


def _clean_name(value) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none"}:
        return ""
    return value


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


def _roster_player_names(roster: pd.DataFrame, keys: set[str]) -> str:
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


def _draft_player_names(drafts: pd.DataFrame, season: int, team: str) -> str:
    d = _draft_rows_for_team(drafts, season, team)
    if d.empty:
        return ""

    name_col = next(
        (c for c in ["pfr_player_name", "player_name", "full_name", "name"] if c in d.columns),
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


def _trade_details(trades: pd.DataFrame, season: int, team: str) -> tuple[float, float, str, str]:
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

    name_col = next(
        (c for c in ["pfr_name", "player_name", "full_name", "name"] if c in t.columns),
        None,
    )

    if name_col:
        player_rows = t[t[name_col].notna()].copy()
    elif "pfr_id" in t.columns:
        player_rows = t[t["pfr_id"].notna()].copy()
    else:
        player_rows = t.copy()

    additions_mask = player_rows.get("received", pd.Series("", index=player_rows.index)) == team
    departures_mask = player_rows.get("gave", pd.Series("", index=player_rows.index)) == team

    additions = float(additions_mask.sum())
    departures = float(departures_mask.sum())

    addition_names = ""
    departure_names = ""
    if name_col:
        addition_names = " | ".join(
            sorted({_clean_name(v) for v in player_rows.loc[additions_mask, name_col] if _clean_name(v)})
        )
        departure_names = " | ".join(
            sorted({_clean_name(v) for v in player_rows.loc[departures_mask, name_col] if _clean_name(v)})
        )

    return additions, departures, addition_names, departure_names


def build_offseason_features(seasons: Iterable[int]) -> pd.DataFrame:
    seasons = sorted(set(int(s) for s in seasons))
    rosters, drafts, trades, player_stats, unavailable = load_offseason_source_data(seasons)
    primary_qbs = _primary_passers(player_stats)

    if rosters.empty:
        return pd.DataFrame(columns=["season", "team"] + OFFSEASON_FEATURES + OFFSEASON_DETAIL_COLUMNS)

    teams = sorted(rosters["team"].dropna().astype(str).unique())
    rows = []

    for season in seasons:
        prior_season = season - 1
        for team in teams:
            prior = rosters[(rosters["season"] == prior_season) & (rosters["team"] == team)]
            current = rosters[(rosters["season"] == season) & (rosters["team"] == team)]
            if prior.empty or current.empty:
                continue

            prior_map = prior.set_index("player_key")["position_weight"].to_dict()
            current_map = current.set_index("player_key")["position_weight"].to_dict()
            prior_keys = set(prior_map)
            current_keys = set(current_map)
            retained = prior_keys & current_keys
            additions = current_keys - prior_keys
            departures = prior_keys - current_keys

            prior_weight = max(sum(prior_map.values()), 1.0)
            current_weight = max(sum(current_map.values()), 1.0)

            continuity = sum(prior_map[k] for k in retained) / prior_weight
            addition_weight = sum(current_map[k] for k in additions) / current_weight
            departure_weight = sum(prior_map[k] for k in departures) / prior_weight

            primary_qb = primary_qbs.get((prior_season, team))
            current_gsis = set(current.get("gsis_id", pd.Series(dtype=str)).dropna().astype(str))
            qb_returning = 1.0 if primary_qb and primary_qb in current_gsis else 0.0

            (
                trade_additions,
                trade_departures,
                trade_added_players,
                trade_departed_players,
            ) = _trade_details(trades, season, team)

            rows.append({
                "season": season,
                "team": team,
                "roster_continuity": float(continuity),
                "addition_weight": float(addition_weight),
                "departure_weight": float(departure_weight),
                "draft_impact": _draft_impact_for_team(drafts, season, team),
                "qb_returning": qb_returning,
                "trade_additions": trade_additions,
                "trade_departures": trade_departures,
                "added_players": _roster_player_names(current, additions),
                "departed_players": _roster_player_names(prior, departures),
                "added_player_details": _roster_player_details(
                    current, additions, player_stats, prior_season
                ),
                "departed_player_details": _roster_player_details(
                    prior, departures, player_stats, prior_season
                ),
                "drafted_players": _draft_player_names(drafts, season, team),
                "trade_added_players": trade_added_players,
                "trade_departed_players": trade_departed_players,
            })

    result = pd.DataFrame(rows)
    result.attrs["unavailable_roster_years"] = unavailable
    return result


def team_offseason_snapshot(offseason: pd.DataFrame, team: str, season: int) -> dict:
    if offseason.empty:
        return {f: 0.0 for f in OFFSEASON_FEATURES}
    row = offseason[(offseason["team"] == team) & (offseason["season"] == season)]
    if row.empty:
        return {f: 0.0 for f in OFFSEASON_FEATURES}
    r = row.iloc[-1]
    return {f: float(r.get(f, 0.0)) for f in OFFSEASON_FEATURES}
