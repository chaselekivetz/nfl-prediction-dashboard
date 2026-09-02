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

OFFENSE_ORDER = [
    "QB", "RB", "FB", "WR", "TE",
    "LT", "LG", "C", "RG", "RT",
]

DEFENSE_ORDER = [
    "LDE", "DE", "NT", "DT", "RDE",
    "LOLB", "LILB", "MLB", "RILB", "ROLB", "LB",
    "LCB", "CB", "NB", "RCB", "FS", "SS", "S",
]

SPECIAL_ORDER = [
    "K", "PK", "P", "PT", "LS", "H", "KR", "PR", "KO", "KOS",
]

POSITION_LABELS = {
    "QB": "Quarterback",
    "RB": "Running Back",
    "FB": "Fullback",
    "WR": "Wide Receiver",
    "TE": "Tight End",
    "LT": "Left Tackle",
    "LG": "Left Guard",
    "C": "Center",
    "RG": "Right Guard",
    "RT": "Right Tackle",
    "LDE": "Left Defensive End",
    "DE": "Defensive End",
    "NT": "Nose Tackle",
    "DT": "Defensive Tackle",
    "RDE": "Right Defensive End",
    "LOLB": "Left Outside Linebacker",
    "LILB": "Left Inside Linebacker",
    "MLB": "Middle Linebacker",
    "RILB": "Right Inside Linebacker",
    "ROLB": "Right Outside Linebacker",
    "LB": "Linebacker",
    "LCB": "Left Cornerback",
    "CB": "Cornerback",
    "NB": "Nickel Corner",
    "RCB": "Right Cornerback",
    "FS": "Free Safety",
    "SS": "Strong Safety",
    "S": "Safety",
    "K": "Kicker",
    "PK": "Kicker",
    "P": "Punter",
    "PT": "Punter",
    "LS": "Long Snapper",
    "H": "Holder",
    "KR": "Kick Returner",
    "PR": "Punt Returner",
    "KO": "Kickoff Specialist",
    "KOS": "Kickoff Specialist",
}


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


def _clean_ourlads_name(value):
    text = _clean(value)
    if not text:
        return ""

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


def _normalize_position(value):
    pos = _clean(value).upper()

    if pos in {"HB"}:
        return "RB"
    if pos in {"LWR", "RWR", "SWR", "SLWR", "SRWR"}:
        return "WR"
    if pos in {"PK"}:
        return "K"
    if pos in {"PT"}:
        return "P"
    if pos in {"KOS"}:
        return "KO"

    return pos


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


def _load_current_depth_chart(team):
    html = _http_get(OURLADS_DEPTH_URL)
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 4:
            continue

        values = [" ".join(cell.stripped_strings).strip() for cell in cells]
        if _norm_team(values[0]) != team:
            continue

        position = _normalize_position(values[1])
        if not position or position in {
            "OFF", "DEF", "ST", "PS", "RES", "IR", "NFI", "PUP"
        }:
            continue

        for rank, index in enumerate(range(3, len(values), 2), start=1):
            name = _clean_ourlads_name(values[index] if index < len(values) else "")
            if not name:
                continue

            number = _clean(values[index - 1] if index - 1 < len(values) else "")
            rows.append(
                {
                    "team": team,
                    "position": position,
                    "rank": rank,
                    "player_name": name,
                    "number": number,
                    "updated": pd.Timestamp.utcnow(),
                    "source": "Current depth chart",
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No current depth-chart rows parsed for {team}.")

    return (
        frame.drop_duplicates(
            ["team", "position", "rank", "player_name"],
            keep="first",
        )
        .sort_values(["position", "rank", "player_name"])
        .reset_index(drop=True)
    )


def _load_nflverse_fallback(season, team):
    try:
        frame = _to_pandas(nfl.load_depth_charts([int(season)]))
    except Exception:
        return pd.DataFrame()

    if frame.empty:
        return frame

    team_col = next(
        (c for c in ["team", "club_code", "club"] if c in frame.columns),
        None,
    )
    name_col = next(
        (c for c in ["player_name", "full_name", "name"] if c in frame.columns),
        None,
    )
    pos_col = next(
        (c for c in ["pos_abb", "position", "pos"] if c in frame.columns),
        None,
    )
    rank_col = next(
        (c for c in ["pos_rank", "depth", "depth_rank"] if c in frame.columns),
        None,
    )
    number_col = next(
        (c for c in ["jersey_number", "jersey", "number"] if c in frame.columns),
        None,
    )

    if not team_col or not name_col or not pos_col:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "team": frame[team_col].map(_norm_team),
            "position": frame[pos_col].map(_normalize_position),
            "rank": (
                pd.to_numeric(frame[rank_col], errors="coerce")
                if rank_col
                else 99
            ),
            "player_name": frame[name_col].map(_clean),
            "number": frame[number_col].map(_clean) if number_col else "",
            "updated": pd.Timestamp.utcnow(),
            "source": "nflverse fallback",
        }
    )

    out = out[(out["team"] == team) & out["player_name"].ne("")].copy()
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(99)

    return (
        out.drop_duplicates(
            ["team", "position", "rank", "player_name"],
            keep="first",
        )
        .sort_values(["position", "rank", "player_name"])
        .reset_index(drop=True)
    )


def load_team_depth_chart(season: int, team: str) -> pd.DataFrame:
    team = _norm_team(team)

    try:
        return _load_current_depth_chart(team)
    except Exception:
        return _load_nflverse_fallback(season, team)


def _position_group(position):
    if position in OFFENSE_ORDER:
        return "OFFENSE"
    if position in DEFENSE_ORDER:
        return "DEFENSE"
    if position in SPECIAL_ORDER:
        return "SPECIAL TEAMS"
    return "OTHER"


def _ordered_positions(rows, group):
    if group == "OFFENSE":
        preferred = OFFENSE_ORDER
    elif group == "DEFENSE":
        preferred = DEFENSE_ORDER
    elif group == "SPECIAL TEAMS":
        preferred = SPECIAL_ORDER
    else:
        preferred = []

    present = list(dict.fromkeys(rows["position"].tolist()))
    ordered = [pos for pos in preferred if pos in present]
    ordered.extend(sorted(pos for pos in present if pos not in ordered))
    return ordered


def _player_chip(row):
    name = escape(_clean(row.player_name))
    number = escape(_clean(row.number))
    rank = int(row.rank) if pd.notna(row.rank) else 99

    rank_label = (
        "Starter"
        if rank == 1
        else f"{rank}"
    )

    number_html = (
        f"<span class='dc-number'>#{number}</span>"
        if number
        else ""
    )

    return (
        "<div class='dc-chip'>"
        f"<span class='dc-rank'>{escape(rank_label)}</span>"
        f"<span class='dc-player-name'>{name}</span>"
        f"{number_html}"
        "</div>"
    )


def depth_chart_html(depth_data: pd.DataFrame, team: str) -> str:
    if depth_data.empty:
        return ""

    rows = depth_data[depth_data["team"] == _norm_team(team)].copy()
    if rows.empty:
        return ""

    rows["rank"] = pd.to_numeric(rows["rank"], errors="coerce").fillna(99)

    sections = []

    for group in ["OFFENSE", "DEFENSE", "SPECIAL TEAMS", "OTHER"]:
        group_rows = rows[rows["position"].map(_position_group) == group].copy()
        if group_rows.empty:
            continue

        position_rows = []
        for position in _ordered_positions(group_rows, group):
            players = (
                group_rows[group_rows["position"] == position]
                .sort_values(["rank", "player_name"])
            )
            chips = "".join(
                _player_chip(row)
                for row in players.itertuples()
            )

            label = POSITION_LABELS.get(position, position)
            position_rows.append(
                "<div class='dc-row'>"
                f"<div class='dc-position'><strong>{escape(position)}</strong>"
                f"<span>{escape(label)}</span></div>"
                f"<div class='dc-players'>{chips}</div>"
                "</div>"
            )

        sections.append(
            "<section class='dc-section'>"
            f"<div class='dc-heading'>{escape(group)}</div>"
            f"{''.join(position_rows)}"
            "</section>"
        )

    return f"""
    <style>
      .dc-simple {{
        font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;
        color:#f7fbff;
      }}

      .dc-section {{
        margin:0 0 18px 0;
        overflow:hidden;
        border:1px solid #22364d;
        border-radius:13px;
        background:linear-gradient(180deg,#07131f,#040a12);
      }}

      .dc-heading {{
        padding:9px 14px;
        font-size:13px;
        font-weight:900;
        letter-spacing:.12em;
        color:#ffffff;
        background:#081a2c;
        border-bottom:1px solid #1e3853;
        border-left:4px solid #0d8cff;
      }}

      .dc-row {{
        display:grid;
        grid-template-columns:170px minmax(0,1fr);
        align-items:center;
        min-height:58px;
        border-bottom:1px solid rgba(255,255,255,.07);
      }}

      .dc-row:last-child {{
        border-bottom:none;
      }}

      .dc-position {{
        height:100%;
        box-sizing:border-box;
        display:flex;
        flex-direction:column;
        justify-content:center;
        padding:8px 13px;
        background:rgba(13,140,255,.06);
        border-right:1px solid rgba(255,255,255,.08);
      }}

      .dc-position strong {{
        color:#8fd3ff;
        font-size:14px;
        line-height:1.1;
      }}

      .dc-position span {{
        color:#7f94aa;
        font-size:10px;
        margin-top:3px;
      }}

      .dc-players {{
        display:flex;
        align-items:center;
        gap:8px;
        padding:8px 10px;
        overflow-x:auto;
        scrollbar-width:thin;
      }}

      .dc-chip {{
        flex:0 0 auto;
        display:flex;
        align-items:center;
        gap:6px;
        min-width:150px;
        padding:8px 10px;
        border-radius:8px;
        background:#0b1725;
        border:1px solid #29425d;
      }}

      .dc-chip:first-child {{
        border-color:#0d8cff;
        box-shadow:inset 0 0 0 1px rgba(13,140,255,.13);
      }}

      .dc-rank {{
        flex:0 0 auto;
        min-width:34px;
        color:#5fc2ff;
        font-size:9px;
        font-weight:800;
        text-transform:uppercase;
      }}

      .dc-player-name {{
        color:#fff;
        font-size:12px;
        font-weight:750;
        white-space:nowrap;
      }}

      .dc-number {{
        color:#7f94aa;
        font-size:9px;
        white-space:nowrap;
      }}

      @media (max-width:850px) {{
        .dc-row {{
          grid-template-columns:110px minmax(0,1fr);
        }}

        .dc-position {{
          padding:7px 8px;
        }}

        .dc-position span {{
          display:none;
        }}

        .dc-chip {{
          min-width:135px;
        }}
      }}
    </style>
    <div class="dc-simple">{''.join(sections)}</div>
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
