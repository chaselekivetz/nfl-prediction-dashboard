from __future__ import annotations

from datetime import datetime
import re

from bs4 import BeautifulSoup
import pandas as pd
import requests
import nflreadpy as nfl


CBS_INJURY_URL = "https://www.cbssports.com/nfl/injuries/"

TEAM_ALIASES = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV",
    "OAK": "LV", "SDG": "LAC", "STL": "LA", "WSH": "WAS",
}

CBS_TEAM_NAMES = {
    "arizona": "ARI",
    "atlanta": "ATL",
    "baltimore": "BAL",
    "buffalo": "BUF",
    "carolina": "CAR",
    "chicago": "CHI",
    "cincinnati": "CIN",
    "cleveland": "CLE",
    "dallas": "DAL",
    "denver": "DEN",
    "detroit": "DET",
    "green bay": "GB",
    "houston": "HOU",
    "indianapolis": "IND",
    "jacksonville": "JAX",
    "kansas city": "KC",
    "las vegas": "LV",
    "l.a. chargers": "LAC",
    "la chargers": "LAC",
    "los angeles chargers": "LAC",
    "l.a. rams": "LA",
    "la rams": "LA",
    "los angeles rams": "LA",
    "miami": "MIA",
    "minnesota": "MIN",
    "new england": "NE",
    "new orleans": "NO",
    "n.y. giants": "NYG",
    "ny giants": "NYG",
    "new york giants": "NYG",
    "n.y. jets": "NYJ",
    "ny jets": "NYJ",
    "new york jets": "NYJ",
    "philadelphia": "PHI",
    "pittsburgh": "PIT",
    "san francisco": "SF",
    "seattle": "SEA",
    "tampa bay": "TB",
    "tennessee": "TEN",
    "washington": "WAS",
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


def _norm_heading(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return value.rstrip(":")


def _team_from_heading(value: str) -> str | None:
    heading = _norm_heading(value)
    if heading in CBS_TEAM_NAMES:
        return CBS_TEAM_NAMES[heading]

    for name, code in CBS_TEAM_NAMES.items():
        if heading == name or heading.endswith(f" {name}") or heading.startswith(f"{name} "):
            return code
    return None


def _cell_text(cell) -> str:
    if cell is None:
        return ""

    # CBS renders both abbreviated and full player names in the player cell.
    # Prefer the longest linked/span value instead of concatenating both.
    candidates = []
    for node in cell.find_all(["a", "span"]):
        text = " ".join(node.stripped_strings).strip()
        if text:
            candidates.append(text)

    if candidates:
        clean = max(candidates, key=len)
        if len(clean) >= 3:
            return clean

    return " ".join(cell.stripped_strings).strip()


def _status_category(detail: str) -> str:
    text = str(detail or "").strip().lower()

    if "questionable" in text:
        return "Questionable"
    if "doubtful" in text:
        return "Doubtful"
    if re.search(r"\bout\b", text):
        return "Out"

    unavailable_terms = [
        "injured reserve",
        "ir.",
        "physically unable to perform",
        "pup",
        "nfi-",
        "non-football injury",
        "suspended",
        "commissioner's exempt",
        "commissioners exempt",
    ]
    if any(term in text for term in unavailable_terms):
        return "Unavailable"

    return "Other"


def _parse_updated(value: str, season: int):
    text = str(value or "").strip()
    if not text:
        return pd.NaT

    # CBS uses strings such as "Tue, Sep 1". Attach the season year.
    parsed = pd.to_datetime(f"{text}, {int(season)}", errors="coerce", utc=True)
    return parsed


def _load_live_cbs_injuries(season: int) -> pd.DataFrame:
    response = requests.get(
        CBS_INJURY_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/149 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=12,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = []

    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        headers = [" ".join(cell.stripped_strings).strip() for cell in header_cells]
        header_text = " | ".join(headers).lower()
        if "player" not in header_text or "injury" not in header_text:
            continue

        heading = table.find_previous(["h2", "h3", "h4", "h5"])
        team = _team_from_heading(" ".join(heading.stripped_strings) if heading else "")
        if not team:
            continue

        body_rows = table.find_all("tr")
        for tr in body_rows:
            cells = tr.find_all("td")
            if len(cells) < 4:
                continue

            # Current CBS layout:
            # Player | Position | Updated | Injury | Injury Status
            player = _cell_text(cells[0])
            position = _cell_text(cells[1]) if len(cells) > 1 else ""
            updated = _cell_text(cells[2]) if len(cells) > 2 else ""
            injury = _cell_text(cells[3]) if len(cells) > 3 else ""
            status_detail = _cell_text(cells[4]) if len(cells) > 4 else ""

            if not player:
                continue

            rows.append(
                {
                    "team": team,
                    "full_name": player,
                    "position": position,
                    "report_primary_injury": injury,
                    "report_secondary_injury": "",
                    "report_status": _status_category(status_detail),
                    "practice_status": "",
                    "status_detail": status_detail,
                    "date_modified": _parse_updated(updated, season),
                    "source": "CBS Sports",
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("CBS injury page returned no parsable injury rows.")

    # A partial/blocked HTML response should not replace the fallback source.
    if frame["team"].nunique() < 16:
        raise RuntimeError(
            f"CBS injury page parsed only {frame['team'].nunique()} teams."
        )

    frame = frame.sort_values(
        ["team", "date_modified"],
        ascending=[True, False],
        na_position="last",
    )
    frame = frame.drop_duplicates(
        ["team", "full_name"],
        keep="first",
    ).reset_index(drop=True)
    frame.attrs["source"] = "CBS Sports live injury tracker"
    frame.attrs["retrieved_at"] = datetime.utcnow().isoformat()
    return frame


def _load_nflverse_injuries(season: int) -> pd.DataFrame:
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

    injuries["status_detail"] = injuries.get(
        "report_status",
        pd.Series("", index=injuries.index),
    ).fillna("").astype(str)
    injuries["source"] = "nflverse"
    injuries.attrs["source"] = "nflverse fallback"
    return injuries


def load_injury_data(season: int) -> pd.DataFrame:
    try:
        return _load_live_cbs_injuries(int(season))
    except Exception as live_error:
        fallback = _load_nflverse_injuries(int(season))
        fallback.attrs["live_source_error"] = str(live_error)
        return fallback


def latest_team_injuries(injuries: pd.DataFrame, team: str) -> pd.DataFrame:
    if injuries.empty or "team" not in injuries.columns:
        return pd.DataFrame()

    team_rows = injuries[injuries["team"] == team].copy()
    if team_rows.empty:
        return team_rows

    # nflverse contains historical weekly reports. The live CBS frame does not
    # use a week column, so it naturally keeps all current statuses.
    if (
        "source" in team_rows.columns
        and not team_rows["source"].astype(str).str.contains("CBS", case=False).any()
        and "week" in team_rows.columns
        and team_rows["week"].notna().any()
    ):
        latest_week = team_rows["week"].max()
        team_rows = team_rows[team_rows["week"] == latest_week].copy()

    if "date_modified" in team_rows.columns:
        team_rows = team_rows.sort_values(
            "date_modified",
            ascending=False,
            na_position="last",
        )

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
            "status_detail",
            "practice_status",
            "date_modified",
            "source",
        ]
        if c in team_rows.columns
    ]
    return team_rows[keep].reset_index(drop=True)


def injury_status_counts(team_injuries: pd.DataFrame) -> dict:
    base = {
        "Out": 0,
        "Doubtful": 0,
        "Questionable": 0,
        "Unavailable": 0,
    }
    if team_injuries.empty or "report_status" not in team_injuries.columns:
        return base

    statuses = (
        team_injuries["report_status"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )
    return {
        "Out": int((statuses == "out").sum()),
        "Doubtful": int((statuses == "doubtful").sum()),
        "Questionable": int((statuses == "questionable").sum()),
        "Unavailable": int((statuses == "unavailable").sum()),
    }
