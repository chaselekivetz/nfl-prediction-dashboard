from __future__ import annotations

from html import escape
import re

from bs4 import BeautifulSoup
import pandas as pd
import requests
import nflreadpy as nfl


OURLADS_DEPTH_URL = "https://www.ourlads.com/nfldepthcharts/depthcharts.aspx"

TEAM_ALIASES = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV",
    "OAK": "LV", "SDG": "LAC", "STL": "LA", "WSH": "WAS",
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

# The field is an original visualization of the published depth chart. Exact
# packages vary by play, so nickel and special-team package roles may add a
# twelfth displayed position.
OFFENSE_LAYOUT = [
    ("LWR", 8, 18), ("SWR", 50, 18), ("RWR", 92, 18),
    ("LT", 18, 51), ("LG", 34, 51), ("C", 50, 51),
    ("RG", 66, 51), ("RT", 82, 51), ("TE", 91, 37),
    ("QB", 50, 70), ("RB", 50, 88),
]

DEFENSE_LAYOUT = [
    ("FS", 39, 12), ("SS", 61, 12),
    ("LOLB", 17, 38), ("LILB", 39, 38), ("RILB", 61, 38), ("ROLB", 83, 38),
    ("DE", 26, 60), ("NT", 50, 60), ("DT", 74, 60),
    ("LCB", 8, 78), ("NB", 50, 87), ("RCB", 92, 78),
]

SPECIAL_LAYOUT = [
    ("PK", 18, 28), ("PT", 39, 28), ("LS", 61, 28), ("H", 82, 28),
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
        name = " ".join(link.stripped_strings).strip()
        if not name:
            continue
        # Remove a jersey number if ESPN's rendered link text appends it.
        name = re.sub(r"(?<=[A-Za-z.)])\d{1,2}$", "", name).strip()
        result[_name_key(name)] = {"espn_id": match.group(1)}

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
        frame = _load_ourlads_team(team)
    except Exception:
        frame = _load_nflverse_fallback(season, team)

    return _enrich_player_ids(frame, season, team)


def _position_candidates(position):
    p = str(position).upper()
    aliases = {
        "LWR": {"LWR", "WR"},
        "SWR": {"SWR", "SLWR", "SRWR", "WR"},
        "RWR": {"RWR", "WR"},
        "TE": {"TE"},
        "LT": {"LT"},
        "LG": {"LG", "OG", "G"},
        "C": {"C"},
        "RG": {"RG", "OG", "G"},
        "RT": {"RT"},
        "QB": {"QB"},
        "RB": {"RB", "HB"},
        "DE": {"DE", "LDE", "RDE"},
        "NT": {"NT", "DT"},
        "DT": {"DT", "RDT", "LDT", "DE"},
        "LOLB": {"LOLB", "OLB", "EDGE", "ED"},
        "LILB": {"LILB", "ILB", "MLB", "LB"},
        "RILB": {"RILB", "ILB", "MLB", "LB"},
        "ROLB": {"ROLB", "OLB", "EDGE", "ED"},
        "LCB": {"LCB", "CB"},
        "RCB": {"RCB", "CB"},
        "NB": {"NB", "CB"},
        "FS": {"FS", "S"},
        "SS": {"SS", "S"},
        "PK": {"PK", "K"},
        "PT": {"PT", "P"},
        "LS": {"LS"},
        "H": {"H", "HLD"},
        "KR": {"KR", "KOR"},
        "PR": {"PR"},
        "KO": {"KO", "KOS", "PK", "K"},
    }
    return aliases.get(p, {p})


def _pool(team_rows, group):
    positions = _position_candidates(group)
    rows = team_rows[team_rows["position"].isin(positions)].copy()

    # Prefer the exact listed position before aliases.
    rows["_exact"] = (rows["position"] == group).astype(int)
    rows = rows.sort_values(["_exact", "rank", "player_name"], ascending=[False, True, True])
    return rows.drop(columns=["_exact"], errors="ignore")


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
    espn_id = _clean(getattr(player, "espn_id", ""))
    if not espn_id:
        return ""
    return f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"


def _card_html(slot, starter, backup, x, y):
    if starter is None:
        return ""

    name = escape(_clean(starter.player_name))
    number = escape(_clean(getattr(starter, "number", "")))
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
    backup_html = f"<div class='dc-backup'>2 · {backup_name}</div>" if backup_name else ""

    return (
        f"<div class='dc-player' style='left:{x}%;top:{y}%;'>"
        f"<div class='dc-pos'>{escape(slot)} {number_html}</div>"
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
    display["Depth"] = display["rank"].map(
        lambda value: "Starter" if int(value) == 1 else f"{int(value)}"
    )
    return display.rename(
        columns={
            "position": "Position",
            "number": "#",
            "player_name": "Player",
        }
    )[["Position", "Depth", "#", "Player"]].reset_index(drop=True)


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
