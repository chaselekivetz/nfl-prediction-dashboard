from __future__ import annotations

from html import escape
import re

from bs4 import BeautifulSoup
import pandas as pd
import requests
import nflreadpy as nfl


OURLADS_DEPTH_URL = "https://www.ourlads.com/nfldepthcharts/depthcharts.aspx"
MADDEN_RATINGS_URL = "https://www.ea.com/games/madden-nfl/ratings"

TEAM_ALIASES = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV",
    "OAK": "LV", "SDG": "LAC", "STL": "LA", "WSH": "WAS",
}

TEAM_FULL_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "NY Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

MADDEN_POSITIONS = {
    "QB","HB","FB","WR","TE","LT","LG","C","RG","RT",
    "LEDG","REDG","DT","NT","MIKE","WILL","SAM","LB",
    "CB","FS","SS","K","P","LS",
}

ESPN_TEAM_SLUGS = {
    "ARI": "ari", "ATL": "atl", "BAL": "bal", "BUF": "buf",
    "CAR": "car", "CHI": "chi", "CIN": "cin", "CLE": "cle",
    "DAL": "dal", "DEN": "den", "DET": "det", "GB": "gb",
    "HOU": "hou", "IND": "ind", "JAX": "jax", "KC": "kc",
    "LV": "lv", "LAC": "lac", "LA": "lar", "MIA": "mia",
    "MIN": "min", "NE": "ne", "NO": "no", "NYG": "nyg",
    "NYJ": "nyj", "PHI": "phi", "PIT": "pit", "SF": "sf",
    "SEA": "sea", "TB": "tb", "TEN": "ten", "WAS": "wsh",
}

# The field is an original Madden-inspired visualization, not a copy of EA's UI.
OFFENSE_LAYOUT = [
    ("WR1", 8, 18), ("WR3", 50, 18), ("WR2", 92, 18),
    ("LT", 18, 51), ("LG", 34, 51), ("C", 50, 51),
    ("RG", 66, 51), ("RT", 82, 51), ("TE", 91, 37),
    ("QB", 50, 70), ("HB", 50, 88),
]

DEFENSE_LAYOUT = [
    ("FS", 39, 12), ("SS", 61, 12),
    ("CB1", 8, 78), ("NCB", 50, 87), ("CB2", 92, 78),
    ("MIKE", 39, 39), ("WILL", 61, 39),
    ("LEDG", 18, 60), ("DT1", 40, 60), ("DT2", 60, 60), ("REDG", 82, 60),
]

SPECIAL_LAYOUT = [
    ("K", 18, 28), ("P", 39, 28), ("LS", 61, 28), ("H", 82, 28),
    ("KR", 27, 70), ("PR", 50, 70), ("KO", 73, 70),
]


def _to_pandas(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def _norm_team(value):
    if pd.isna(value):
        return ""
    code = str(value).upper().strip()
    return TEAM_ALIASES.get(code, code)


def _clean(value):
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).strip())
    return "" if text.lower() in {"nan", "none", "-"} else text


def _name_key(value):
    text = _clean(value).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    parts = [p for p in text.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return " ".join(parts)


def _clean_ourlads_name(value):
    text = _clean(value)
    if not text:
        return ""

    # Ourlads appends acquisition/draft shorthand such as T/Cle, U/KC,
    # CF26, 24/3, etc. Strip those display-only tags.
    previous = None
    while previous != text:
        previous = text
        text = re.sub(
            r"\s+(?:(?:CC|CF|SF|T|U|W|P)/?[A-Za-z0-9]+|\d{2}/\d+|[A-Za-z]{1,3}\d{2})\*?\^?$",
            "",
            text,
            flags=re.I,
        ).strip()

    if "," in text:
        last, first = [part.strip() for part in text.split(",", 1)]
        if first and last:
            text = f"{first} {last}"

    return re.sub(r"\s+", " ", text).strip().title()


def _http_get(url):
    response = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/149 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.text


def _discover_madden_team_url(team: str) -> str:
    target = _clean(TEAM_FULL_NAMES.get(team, "")).lower()
    if not target:
        return ""

    try:
        soup = BeautifulSoup(_http_get(MADDEN_RATINGS_URL), "html.parser")
    except Exception:
        return ""

    candidates = []
    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        if "/ratings/teams-ratings/" not in href:
            continue
        label = _clean(" ".join(link.stripped_strings)).lower()
        if href.startswith("/"):
            href = "https://www.ea.com" + href
        candidates.append((label, href))

    aliases = {target}
    if team == "NYJ":
        aliases.update({"new york jets", "ny jets"})
    if team == "LA":
        aliases.update({"los angeles rams", "la rams"})
    if team == "LAC":
        aliases.update({"los angeles chargers", "la chargers"})

    for label, href in candidates:
        if label in aliases:
            return href
    for label, href in candidates:
        if any(alias in label or label in alias for alias in aliases):
            return href
    return ""


def _load_madden_team(team: str) -> pd.DataFrame:
    url = _discover_madden_team_url(team)

    # Reliable fallback for the Rams if EA's ratings index only exposes a
    # subset of team links in its initial HTML.
    if not url and team == "LA":
        url = (
            "https://www.ea.com/games/madden-nfl/ratings/"
            "teams-ratings/los-angeles-rams/24"
        )

    if not url:
        return pd.DataFrame()

    soup = BeautifulSoup(_http_get(url), "html.parser")
    rows = []

    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        if "/ratings/player-ratings/" not in href:
            continue

        name = _clean(" ".join(link.stripped_strings))
        tr = link.find_parent("tr")
        if not name or tr is None:
            continue

        row_text = " ".join(tr.stripped_strings)
        ovr_match = re.search(r"\bOVR\s*(\d{2})\b", row_text, flags=re.I)
        if not ovr_match:
            continue

        position = ""
        for node in tr.find_all(["a", "span", "div", "td"]):
            text = _clean(" ".join(node.stripped_strings)).upper()
            if text in MADDEN_POSITIONS:
                position = text
                break

        if not position:
            # Fallback: choose a compact all-caps token near the player name.
            tokens = re.findall(r"\b[A-Z]{1,5}\b", row_text.upper())
            position = next((t for t in tokens if t in MADDEN_POSITIONS), "")

        if not position:
            continue

        rows.append(
            {
                "team": team,
                "position": position,
                "rank": 99,
                "player_name": name,
                "number": "",
                "espn_id": "",
                "headshot_url": "",
                "madden_ovr": int(ovr_match.group(1)),
                "updated": pd.Timestamp.utcnow(),
                "source": "EA SPORTS Madden NFL 27",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame = frame.drop_duplicates(["player_name"], keep="first")
    return frame.sort_values(
        ["madden_ovr", "position", "player_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _load_ourlads_team(team: str) -> pd.DataFrame:
    html = _http_get(OURLADS_DEPTH_URL)
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue

        values = [" ".join(cell.stripped_strings).strip() for cell in cells]
        row_team = _norm_team(values[0])
        if row_team != team:
            continue

        position = _clean(values[1]).upper()
        if not position or position in {"OFF", "DEF", "ST", "PS", "RES", "IR", "NFI", "PUP"}:
            continue

        # Layout: Team | Pos | No | Player1 | No | Player2 ...
        for rank, index in enumerate(range(3, len(values), 2), start=1):
            raw_name = values[index] if index < len(values) else ""
            name = _clean_ourlads_name(raw_name)
            if not name:
                continue
            number = values[index - 1] if index - 1 < len(values) else ""
            rows.append(
                {
                    "team": team,
                    "position": position,
                    "rank": rank,
                    "player_name": name,
                    "number": _clean(number),
                    "espn_id": "",
                    "headshot_url": "",
                    "madden_ovr": pd.NA,
                    "updated": pd.Timestamp.utcnow(),
                    "source": "Current depth chart",
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No current depth-chart rows parsed for {team}.")

    return frame.drop_duplicates(
        ["team", "position", "rank", "player_name"]
    ).reset_index(drop=True)


def _load_nflverse_roster_ids(season: int, team: str) -> dict:
    try:
        roster = _to_pandas(nfl.load_rosters([int(season)]))
    except Exception:
        return {}

    if roster.empty or "team" not in roster.columns:
        return {}

    roster = roster.copy()
    roster["team"] = roster["team"].map(_norm_team)
    roster = roster[roster["team"] == team].copy()
    if roster.empty:
        return {}

    name_col = next(
        (c for c in ["full_name", "player_name", "name"] if c in roster.columns),
        None,
    )
    espn_col = next(
        (c for c in ["espn_id", "espn_player_id"] if c in roster.columns),
        None,
    )
    number_col = next(
        (c for c in ["jersey_number", "jersey", "number"] if c in roster.columns),
        None,
    )
    headshot_col = next(
        (
            c
            for c in ["headshot_url", "headshot", "headshot_href"]
            if c in roster.columns
        ),
        None,
    )
    if not name_col:
        return {}

    result = {}
    for row in roster.itertuples():
        name = _clean(getattr(row, name_col, ""))
        if not name:
            continue
        result[_name_key(name)] = {
            "espn_id": _clean(getattr(row, espn_col, "")) if espn_col else "",
            "number": _clean(getattr(row, number_col, "")) if number_col else "",
            "headshot_url": (
                _clean(getattr(row, headshot_col, "")) if headshot_col else ""
            ),
        }
    return result


def _load_espn_roster_ids(team: str) -> dict:
    slug = ESPN_TEAM_SLUGS.get(team)
    if not slug:
        return {}

    try:
        html = _http_get(f"https://www.espn.com/nfl/team/roster/_/name/{slug}")
    except Exception:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    result = {}
    player_re = re.compile(r"/nfl/player/_/id/(\d+)")

    for link in soup.find_all("a", href=True):
        match = player_re.search(link.get("href", ""))
        if not match:
            continue

        name = _clean(" ".join(link.stripped_strings))
        if not name:
            continue
        name = re.sub(r"(?<=[A-Za-z.)])\d{1,2}$", "", name).strip()

        headshot = ""
        row = link.find_parent("tr")
        if row is not None:
            img = row.find("img")
            if img is not None:
                headshot = _clean(
                    img.get("src")
                    or img.get("data-src")
                    or img.get("data-default-src")
                )

        result[_name_key(name)] = {
            "espn_id": match.group(1),
            "headshot_url": headshot,
        }

    return result


def _enrich_player_ids(frame: pd.DataFrame, season: int, team: str) -> pd.DataFrame:
    if frame.empty:
        return frame

    nflverse_ids = _load_nflverse_roster_ids(season, team)
    espn_ids = _load_espn_roster_ids(team)

    out = frame.copy()
    for index, row in out.iterrows():
        key = _name_key(row["player_name"])
        nfl_info = nflverse_ids.get(key, {})
        espn_info = espn_ids.get(key, {})

        espn_id = _clean(nfl_info.get("espn_id")) or _clean(espn_info.get("espn_id"))
        if espn_id:
            out.at[index, "espn_id"] = espn_id

        headshot = (
            _clean(nfl_info.get("headshot_url"))
            or _clean(espn_info.get("headshot_url"))
        )
        if headshot:
            out.at[index, "headshot_url"] = headshot

        if not _clean(row.get("number")):
            number = _clean(nfl_info.get("number"))
            if number:
                out.at[index, "number"] = number

    return out


def _load_nflverse_fallback(season: int, team: str) -> pd.DataFrame:
    try:
        frame = _to_pandas(nfl.load_depth_charts([int(season)]))
    except Exception:
        return pd.DataFrame()

    if frame.empty:
        return frame

    team_col = next((c for c in ["team", "club_code", "club"] if c in frame.columns), None)
    name_col = next((c for c in ["player_name", "full_name", "name"] if c in frame.columns), None)
    pos_col = next((c for c in ["pos_abb", "position", "pos"] if c in frame.columns), None)
    rank_col = next((c for c in ["pos_rank", "depth", "depth_rank"] if c in frame.columns), None)
    number_col = next((c for c in ["jersey_number", "jersey", "number"] if c in frame.columns), None)
    espn_col = next((c for c in ["espn_id", "espn_player_id"] if c in frame.columns), None)

    if not team_col or not name_col or not pos_col:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "team": frame[team_col].map(_norm_team),
            "player_name": frame[name_col].map(_clean),
            "position": frame[pos_col].map(_clean).str.upper(),
            "rank": pd.to_numeric(frame[rank_col], errors="coerce") if rank_col else 99,
            "number": frame[number_col].map(_clean) if number_col else "",
            "espn_id": frame[espn_col].map(_clean) if espn_col else "",
            "headshot_url": "",
            "madden_ovr": pd.NA,
            "updated": pd.Timestamp.utcnow(),
            "source": "nflverse fallback",
        }
    )
    out = out[(out["team"] == team) & out["player_name"].ne("")].copy()
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(99)
    return out.sort_values(["position", "rank", "player_name"]).reset_index(drop=True)


def load_team_depth_chart(season: int, team: str) -> pd.DataFrame:
    team = _norm_team(team)

    try:
        current = _load_ourlads_team(team)
    except Exception:
        current = _load_nflverse_fallback(season, team)

    try:
        madden = _load_madden_team(team)
    except Exception:
        madden = pd.DataFrame()

    if madden.empty:
        return _enrich_player_ids(current, season, team)

    # Madden determines the starter ordering. Current depth data supplements
    # specialist roles and late players that are not present in the ratings DB.
    current_keys = set()
    if not current.empty:
        current = current.copy()
        current["_name_key"] = current["player_name"].map(_name_key)
        current_keys = set(current["_name_key"])

    madden = madden.copy()
    madden["_name_key"] = madden["player_name"].map(_name_key)
    madden_keys = set(madden["_name_key"])

    supplement = pd.DataFrame()
    if not current.empty:
        specialist_positions = {"KR", "PR", "KO", "KOS", "H", "HLD", "LS"}
        supplement = current[
            (~current["_name_key"].isin(madden_keys))
            | (current["position"].isin(specialist_positions))
        ].copy()

    merged = pd.concat([madden, supplement], ignore_index=True, sort=False)
    merged = merged.drop(columns=["_name_key"], errors="ignore")
    merged["madden_ovr"] = pd.to_numeric(
        merged.get("madden_ovr"),
        errors="coerce",
    )
    merged["rank"] = pd.to_numeric(merged.get("rank"), errors="coerce").fillna(99)

    merged = _enrich_player_ids(merged, season, team)
    return merged.drop_duplicates(
        ["team", "position", "player_name"],
        keep="first",
    ).reset_index(drop=True)


def _position_candidates(position):
    p = str(position).upper()
    aliases = {
        "WR1": {"WR"}, "WR2": {"WR"}, "WR3": {"WR"},
        "TE": {"TE"}, "LT": {"LT"}, "LG": {"LG"}, "C": {"C"},
        "RG": {"RG"}, "RT": {"RT"}, "QB": {"QB"}, "HB": {"HB", "RB"},
        "LEDG": {"LEDG", "LE", "DE", "EDGE"},
        "REDG": {"REDG", "RE", "DE", "EDGE"},
        "DT1": {"DT", "NT"}, "DT2": {"DT", "NT"},
        "MIKE": {"MIKE", "MLB", "ILB", "LB"},
        "WILL": {"WILL", "SAM", "OLB", "LB"},
        "CB1": {"CB"}, "CB2": {"CB"}, "NCB": {"CB", "NB"},
        "FS": {"FS"}, "SS": {"SS"},
        "K": {"K", "PK"}, "P": {"P", "PT"}, "LS": {"LS"},
        "H": {"H", "HLD"}, "KR": {"KR", "KOR"}, "PR": {"PR"},
        "KO": {"KO", "KOS", "K", "PK"},
    }
    return aliases.get(p, {p})


def _pool(team_rows, group):
    positions = _position_candidates(group)
    rows = team_rows[team_rows["position"].isin(positions)].copy()
    if rows.empty:
        return rows

    exact_position = group
    if group in {"WR1", "WR2", "WR3"}:
        exact_position = "WR"
    elif group in {"CB1", "CB2", "NCB"}:
        exact_position = "CB"
    elif group in {"DT1", "DT2"}:
        exact_position = "DT"

    rows["_exact"] = (rows["position"] == exact_position).astype(int)
    rows["_ovr"] = pd.to_numeric(rows.get("madden_ovr"), errors="coerce").fillna(-1)
    rows["_depth_rank"] = pd.to_numeric(rows.get("rank"), errors="coerce").fillna(99)

    rows = rows.sort_values(
        ["_exact", "_ovr", "_depth_rank", "player_name"],
        ascending=[False, False, True, True],
    )
    return rows.drop(columns=["_exact", "_ovr", "_depth_rank"], errors="ignore")


def _slot_payload(slot, team_rows, used):
    rows = _pool(team_rows, slot)
    starter = None
    for row in rows.itertuples():
        key = _name_key(row.player_name)
        if key in used:
            continue
        starter = row
        used.add(key)
        break

    if starter is None:
        return None, None

    backup = None
    for row in rows.itertuples():
        if _name_key(row.player_name) == _name_key(starter.player_name):
            continue
        backup = row
        break

    return starter, backup


def _headshot_url(player):
    if player is None:
        return ""

    direct = _clean(getattr(player, "headshot_url", ""))
    if direct:
        return direct

    espn_id = _clean(getattr(player, "espn_id", ""))
    if not espn_id:
        return ""
    return f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"


def _card_html(slot, starter, backup, x, y):
    if starter is None:
        return ""

    name = escape(_clean(starter.player_name))
    number = escape(_clean(getattr(starter, "number", "")))
    madden_ovr = pd.to_numeric(
        getattr(starter, "madden_ovr", pd.NA),
        errors="coerce",
    )
    headshot = _headshot_url(starter)
    backup_name = escape(_clean(backup.player_name)) if backup is not None else ""

    face_html = (
        f"<img class='dc-face' src='{escape(headshot)}' alt='' loading='lazy' "
        f"onerror=\"this.style.display='none';this.nextElementSibling.style.display='flex'\"/>"
        f"<div class='dc-face dc-face-fallback' style='display:none'>"
        f"{escape(name[:1].upper())}</div>"
        if headshot
        else f"<div class='dc-face dc-face-fallback'>{escape(name[:1].upper())}</div>"
    )
    number_html = f"<span class='dc-number'>#{number}</span>" if number else ""
    ovr_html = (
        f"<span class='dc-ovr'>{int(madden_ovr)} OVR</span>"
        if pd.notna(madden_ovr)
        else ""
    )
    backup_html = f"<div class='dc-backup'>2 · {backup_name}</div>" if backup_name else ""

    return (
        f"<div class='dc-player' style='left:{x}%;top:{y}%;'>"
        f"<div class='dc-pos'>{escape(slot)} {number_html} {ovr_html}</div>"
        f"{face_html}"
        f"<div class='dc-name'>{name}</div>"
        f"{backup_html}"
        f"</div>"
    )


def _field_html(title, layout, rows):
    used = set()
    cards = []
    for slot, x, y in layout:
        starter, backup = _slot_payload(slot, rows, used)
        card = _card_html(slot, starter, backup, x, y)
        if card:
            cards.append(card)

    return (
        f"<section class='dc-panel'>"
        f"<div class='dc-panel-title'>{escape(title)}</div>"
        f"<div class='dc-field'>"
        f"<div class='dc-midline'></div>"
        f"<div class='dc-yard y1'></div><div class='dc-yard y2'></div>"
        f"<div class='dc-yard y3'></div><div class='dc-yard y4'></div>"
        f"{''.join(cards)}"
        f"</div></section>"
    )


def depth_chart_html(depth_data: pd.DataFrame, team: str) -> str:
    if depth_data.empty:
        return ""

    rows = depth_data[depth_data["team"] == _norm_team(team)].copy()
    if rows.empty:
        return ""

    offense = _field_html("OFFENSE", OFFENSE_LAYOUT, rows)
    defense = _field_html("DEFENSE", DEFENSE_LAYOUT, rows)
    special = _field_html("SPECIAL TEAMS", SPECIAL_LAYOUT, rows)

    return f"""
    <style>
      .dc-wrap {{
        --blue:#087cff;--cyan:#36c8ff;--muted:#8fa4be;
        font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;
      }}
      .dc-grid {{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}
      .dc-panel {{
        background:linear-gradient(180deg,#071522,#030914);
        border:1px solid #22354c;border-radius:14px;padding:9px;
        box-shadow:inset 0 0 0 1px rgba(8,124,255,.12);
        min-width:0;
      }}
      .dc-panel-title {{
        color:white;font-weight:900;letter-spacing:.12em;font-size:14px;
        border-left:4px solid var(--blue);padding-left:9px;margin:1px 0 7px 2px;
      }}
      .dc-field {{
        position:relative;height:390px;overflow:hidden;border-radius:12px;
        background:
          linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),
          linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px),
          radial-gradient(circle at 50% 50%,rgba(8,124,255,.10),transparent 48%),#07131b;
        background-size:100% 43px,58px 100%,100% 100%,100% 100%;
        border:1px solid #24384a;
      }}
      .dc-field:before,.dc-field:after {{
        content:"";position:absolute;left:5%;right:5%;height:1px;background:rgba(255,255,255,.12)
      }}
      .dc-field:before {{top:24%}} .dc-field:after {{top:76%}}
      .dc-midline {{
        position:absolute;left:3%;right:3%;top:52%;height:2px;
        background:#087cff;box-shadow:0 0 12px #087cff
      }}
      .dc-yard {{position:absolute;top:0;bottom:0;width:1px;background:rgba(255,255,255,.06)}}
      .dc-yard.y1{{left:20%}} .dc-yard.y2{{left:40%}} .dc-yard.y3{{left:60%}} .dc-yard.y4{{left:80%}}
      .dc-player {{
        position:absolute;transform:translate(-50%,-50%);
        width:72px;min-height:58px;padding:3px 3px;border-radius:8px;
        background:linear-gradient(180deg,#0d1d31,#07101c);
        border:1px solid #4a6078;
        box-shadow:0 4px 12px rgba(0,0,0,.35),inset 0 0 0 1px rgba(20,149,255,.15);
        text-align:center;color:#fff;overflow:hidden;
      }}
      .dc-pos {{font-size:8px;font-weight:900;color:#cfe8ff;letter-spacing:.04em;line-height:1}}
      .dc-number {{font-size:7px;color:#72bfff}}
      .dc-ovr {{font-size:6.5px;color:#59d0ff;margin-left:2px}}
      .dc-face {{
        width:25px;height:25px;border-radius:50%;object-fit:cover;object-position:center 12%;
        margin:2px auto 1px;background:#0f2238;border:1px solid rgba(54,200,255,.45)
      }}
      .dc-face-fallback {{
        display:flex;align-items:center;justify-content:center;color:#d8ecff;
        font-size:11px;font-weight:900;
        background:radial-gradient(circle at 50% 38%,#52708d 0 25%,#1a3149 26% 55%,#0d1b2a 56%)
      }}
      .dc-name {{
        font-size:7.5px;font-weight:850;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.1
      }}
      .dc-backup {{
        font-size:5.8px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
        margin-top:1px;border-top:1px solid rgba(255,255,255,.08);padding-top:1px;line-height:1.05
      }}
      @media (max-width:1150px) {{
        .dc-grid {{grid-template-columns:1fr}}
        .dc-field {{height:390px}}
        .dc-player {{width:78px}}
      }}
    </style>
    <div class="dc-wrap"><div class="dc-grid">{offense}{defense}{special}</div></div>
    """


def full_depth_table(depth_data: pd.DataFrame) -> pd.DataFrame:
    if depth_data.empty:
        return pd.DataFrame()

    display = depth_data.copy()
    display["Madden OVR"] = pd.to_numeric(
        display.get("madden_ovr"),
        errors="coerce",
    ).astype("Int64")
    display["Depth"] = display["rank"].map(
        lambda value: "Madden-ranked" if int(value) == 99 else (
            "Starter" if int(value) == 1 else f"{int(value)}"
        )
    )
    return display.rename(
        columns={
            "position": "Position",
            "number": "#",
            "player_name": "Player",
            "source": "Source",
        }
    )[["Position", "Depth", "#", "Player", "Madden OVR", "Source"]].reset_index(drop=True)


def latest_update_label(depth_data: pd.DataFrame, team: str) -> str:
    if depth_data.empty or "updated" not in depth_data.columns:
        return ""
    values = depth_data["updated"].dropna()
    if values.empty:
        return ""
    try:
        return values.max().strftime("%b %d, %Y")
    except Exception:
        return ""
