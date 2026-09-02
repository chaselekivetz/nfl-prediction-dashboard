from __future__ import annotations

from html import escape
import re

from bs4 import BeautifulSoup
import pandas as pd
import requests
import nflreadpy as nfl


OURLADS_DEPTH_URL = "https://www.ourlads.com/nfldepthcharts/depthcharts.aspx"

TEAM_ALIASES = {
    "ARZ": "ARI", "GNB": "GB", "KAN": "KC", "LAR": "LA", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV",
    "OAK": "LV", "SDG": "LAC", "STL": "LA", "WSH": "WAS",
}

OFFENSE_ORDER = [
    "QB", "RB", "FB",
    "LWR", "RWR", "SWR", "WR",
    "TE", "LT", "LG", "C", "RG", "RT", "OT", "OG",
]

DEFENSE_ORDER = [
    "LDE", "DE", "NT", "DT", "RDE", "ED", "EDGE",
    "LOLB", "WLB", "LILB", "MLB", "ILB", "RILB", "SLB", "ROLB", "LB",
    "LCB", "CB", "NB", "RCB", "FS", "SS", "S",
]

SPECIAL_ORDER = [
    "PT", "P", "PK", "K", "LS", "H", "KO", "KOS", "PR", "KR",
]

EXPECTED_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LV", "LAC", "LA", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SF", "SEA", "TB", "TEN", "WAS",
}

POSITION_LABELS = {
    "QB": "Quarterback",
    "RB": "Running Back",
    "FB": "Fullback",
    "LWR": "Wide Receiver",
    "RWR": "Wide Receiver",
    "SWR": "Wide Receiver",
    "WR": "Wide Receiver",
    "TE": "Tight End",
    "LT": "Left Tackle",
    "LG": "Left Guard",
    "C": "Center",
    "RG": "Right Guard",
    "RT": "Right Tackle",
    "OT": "Offensive Tackle",
    "OG": "Offensive Guard",
    "LDE": "Defensive End",
    "DE": "Defensive End",
    "NT": "Nose Tackle",
    "DT": "Defensive Tackle",
    "RDE": "Defensive End",
    "ED": "Edge",
    "EDGE": "Edge",
    "LOLB": "Outside Linebacker",
    "WLB": "Weakside Linebacker",
    "LILB": "Inside Linebacker",
    "MLB": "Middle Linebacker",
    "ILB": "Inside Linebacker",
    "RILB": "Inside Linebacker",
    "SLB": "Strongside Linebacker",
    "ROLB": "Outside Linebacker",
    "LB": "Linebacker",
    "LCB": "Cornerback",
    "CB": "Cornerback",
    "NB": "Nickel Corner",
    "RCB": "Cornerback",
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


def _display_position(position):
    position = _clean(position).upper()
    aliases = {
        "LWR": "WR",
        "RWR": "WR",
        "SWR": "WR",
        "LDE": "DE",
        "RDE": "DE",
        "LCB": "CB",
        "RCB": "CB",
        "LILB": "ILB",
        "RILB": "ILB",
        "LOLB": "OLB",
        "ROLB": "OLB",
        "PT": "P",
        "PK": "K",
    }
    return aliases.get(position, position)



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
    return _clean(value).upper()


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


def _load_current_depth_charts_all():
    html = _http_get(OURLADS_DEPTH_URL)
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    current_section = None
    slot_order = {}

    for tr in soup.find_all("tr"):
        row_text = _clean(" ".join(tr.stripped_strings))
        lowered = row_text.lower()

        if lowered.startswith("offense -"):
            current_section = "OFFENSE"
            continue
        if lowered.startswith("defense -"):
            current_section = "DEFENSE"
            continue
        if lowered.startswith("special teams -"):
            current_section = "SPECIAL TEAMS"
            continue
        if (
            lowered.startswith("practice squad -")
            or lowered.startswith("reserves -")
        ):
            current_section = None
            continue

        if current_section is None:
            continue

        # Only use direct table cells. Nested markup can shift depth columns.
        cells = tr.find_all(["td", "th"], recursive=False)
        if len(cells) < 4:
            continue

        values = [" ".join(cell.stripped_strings).strip() for cell in cells]
        team = _norm_team(values[0])
        if team not in EXPECTED_TEAMS:
            continue

        position = _normalize_position(values[1])
        if not position or position in {"OFF", "DEF", "ST"}:
            continue

        team_slot_key = (team, current_section)
        slot_order[team_slot_key] = slot_order.get(team_slot_key, 0) + 1
        source_slot_order = slot_order[team_slot_key]
        slot_id = f"{current_section}:{source_slot_order}:{position}"

        player_slots = [
            (1, 2, 3),
            (2, 4, 5),
            (3, 6, 7),
            (4, 8, 9),
            (5, 10, 11),
        ]

        added_player = False
        for rank, number_index, name_index in player_slots:
            if name_index >= len(values):
                continue

            name = _clean_ourlads_name(values[name_index])
            if not name:
                continue

            number = _clean(
                values[number_index]
                if number_index < len(values)
                else ""
            )

            rows.append(
                {
                    "team": team,
                    "section": current_section,
                    "position": position,
                    "slot_id": slot_id,
                    "slot_order": source_slot_order,
                    "rank": rank,
                    "player_name": name,
                    "number": number,
                    "updated": pd.Timestamp.utcnow(),
                    "source": "Current depth chart",
                }
            )
            added_player = True

        if not added_player:
            rows.append(
                {
                    "team": team,
                    "section": current_section,
                    "position": position,
                    "slot_id": slot_id,
                    "slot_order": source_slot_order,
                    "rank": 1,
                    "player_name": "TBD",
                    "number": "",
                    "updated": pd.Timestamp.utcnow(),
                    "source": "Current depth chart",
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No current depth-chart rows were parsed.")

    parsed_teams = set(frame["team"].dropna().unique())
    if len(parsed_teams) < 32:
        missing = sorted(EXPECTED_TEAMS - parsed_teams)
        raise RuntimeError(
            "Current depth-chart source was incomplete. "
            f"Parsed {len(parsed_teams)} teams; missing: {', '.join(missing)}"
        )

    return (
        frame.drop_duplicates(
            ["team", "slot_id", "rank", "player_name"],
            keep="first",
        )
        .sort_values(["team", "section", "slot_order", "rank"])
        .reset_index(drop=True)
    )


def _load_current_depth_chart(team):
    frame = _load_current_depth_charts_all()
    team_rows = frame[frame["team"] == team].copy()
    if team_rows.empty:
        raise RuntimeError(f"No current depth-chart rows parsed for {team}.")
    return team_rows.reset_index(drop=True)


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
    out["section"] = out["position"].map(_position_group)
    out["slot_order"] = (
        out.groupby("section", dropna=False)["position"]
        .transform(lambda values: pd.factorize(values)[0] + 1)
    )
    out["slot_id"] = (
        out["section"].astype(str)
        + ":"
        + out["slot_order"].astype(str)
        + ":"
        + out["position"].astype(str)
    )

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
    position = _clean(position).upper()
    if position in set(OFFENSE_ORDER):
        return "OFFENSE"
    if position in set(DEFENSE_ORDER):
        return "DEFENSE"
    if position in set(SPECIAL_ORDER):
        return "SPECIAL TEAMS"

    if position in {"HB", "T", "G"}:
        return "OFFENSE"
    if position in {"DL", "DB"}:
        return "DEFENSE"
    return "OTHER"


def _ordered_slots(rows, group):
    if rows.empty:
        return []

    preferred = (
        OFFENSE_ORDER
        if group == "OFFENSE"
        else DEFENSE_ORDER
        if group == "DEFENSE"
        else SPECIAL_ORDER
        if group == "SPECIAL TEAMS"
        else []
    )
    priority = {position: index for index, position in enumerate(preferred)}

    slots = (
        rows[["slot_id", "slot_order", "position"]]
        .drop_duplicates("slot_id")
        .copy()
    )
    slots["_priority"] = slots["position"].map(
        lambda value: priority.get(value, 999)
    )
    slots = slots.sort_values(["_priority", "slot_order", "position"])

    return list(slots.itertuples(index=False))


DEPTH_COLUMN_LABELS = {
    1: "Starter",
    2: "Backup",
    3: "3rd String",
    4: "4th String",
    5: "5th String",
}


def _player_cell(row):
    if row is None:
        return "<div class='dc-player-cell dc-empty'></div>"

    name = escape(_clean(row.player_name))
    number = escape(_clean(row.number))
    number_html = (
        f"<span class='dc-number'>#{number}</span>"
        if number
        else ""
    )

    return (
        "<div class='dc-player-cell'>"
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

    if "section" in rows.columns:
        row_groups = rows["section"].fillna(
            rows["position"].map(_position_group)
        )
    else:
        row_groups = rows["position"].map(_position_group)

    for group in ["OFFENSE", "DEFENSE", "SPECIAL TEAMS", "OTHER"]:
        group_rows = rows[row_groups == group].copy()
        if group_rows.empty:
            continue

        position_rows = []
        for slot in _ordered_slots(group_rows, group):
            position = _clean(slot.position).upper()
            players = (
                group_rows[group_rows["slot_id"] == slot.slot_id]
                .sort_values(["rank", "player_name"])
            )

            by_rank = {}
            for player in players.itertuples():
                rank = int(player.rank) if pd.notna(player.rank) else 99
                if 1 <= rank <= 5 and rank not in by_rank:
                    by_rank[rank] = player

            player_cells = "".join(
                _player_cell(by_rank.get(rank))
                for rank in range(1, 6)
            )

            display_position = _display_position(position)
            label = POSITION_LABELS.get(position, position)
            position_rows.append(
                "<div class='dc-row'>"
                f"<div class='dc-position'><strong>{escape(display_position)}</strong>"
                f"<span>{escape(label)}</span></div>"
                f"{player_cells}"
                "</div>"
            )

        header_cells = "".join(
            f"<div class='dc-depth-label'>{escape(DEPTH_COLUMN_LABELS[rank])}</div>"
            for rank in range(1, 6)
        )

        sections.append(
            "<section class='dc-section'>"
            f"<div class='dc-heading'>{escape(group)}</div>"
            "<div class='dc-columns'>"
            "<div class='dc-position-header'>Position</div>"
            f"{header_cells}"
            "</div>"
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
        overflow-x:auto;
        overflow-y:hidden;
        border:1px solid #22364d;
        border-radius:13px;
        background:linear-gradient(180deg,#07131f,#040a12);
      }}

      .dc-heading {{
        min-width:900px;
        padding:9px 14px;
        box-sizing:border-box;
        font-size:13px;
        font-weight:900;
        letter-spacing:.12em;
        color:#ffffff;
        background:#081a2c;
        border-bottom:1px solid #1e3853;
        border-left:4px solid #0d8cff;
      }}

      .dc-columns,
      .dc-row {{
        display:grid;
        grid-template-columns:150px repeat(5, 150px);
        width:900px;
        align-items:stretch;
      }}

      .dc-columns {{
        background:#08131f;
        border-bottom:1px solid rgba(255,255,255,.1);
      }}

      .dc-position-header,
      .dc-depth-label {{
        padding:7px 10px;
        color:#7f94aa;
        font-size:9px;
        font-weight:800;
        text-transform:uppercase;
        letter-spacing:.06em;
        border-right:1px solid rgba(255,255,255,.06);
      }}

      .dc-row {{
        min-height:50px;
        border-bottom:1px solid rgba(255,255,255,.07);
      }}

      .dc-row:last-child {{
        border-bottom:none;
      }}

      .dc-position {{
        display:flex;
        flex-direction:column;
        justify-content:center;
        box-sizing:border-box;
        padding:7px 10px;
        background:rgba(13,140,255,.06);
        border-right:1px solid rgba(255,255,255,.08);
      }}

      .dc-position strong {{
        color:#8fd3ff;
        font-size:13px;
        line-height:1.1;
      }}

      .dc-position span {{
        color:#7f94aa;
        font-size:9px;
        margin-top:2px;
      }}

      .dc-player-cell {{
        display:flex;
        align-items:center;
        gap:5px;
        padding:7px 9px;
        box-sizing:border-box;
        border-right:1px solid rgba(255,255,255,.06);
      }}

      .dc-empty {{
        background:transparent;
      }}

      .dc-player-name {{
        color:#ffffff;
        font-size:11px;
        font-weight:750;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
        max-width:118px;
      }}

      .dc-number {{
        color:#71869d;
        font-size:8px;
        white-space:nowrap;
      }}

      @media (max-width:850px) {{
        .dc-columns,
        .dc-row {{
          grid-template-columns:105px repeat(5, 140px);
          width:805px;
        }}

        .dc-heading {{
          min-width:805px;
        }}

        .dc-position {{
          padding:6px 7px;
        }}

        .dc-position span {{
          display:none;
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
