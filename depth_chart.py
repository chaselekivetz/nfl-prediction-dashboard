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

OURLADS_TEAM_CODES = {
    "ARI": "ARZ", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB": "GB",
    "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC": "KC",
    "LV": "LV", "LAC": "LAC", "LA": "LAR", "MIA": "MIA",
    "MIN": "MIN", "NE": "NE", "NO": "NO", "NYG": "NYG",
    "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SF": "SF",
    "SEA": "SEA", "TB": "TB", "TEN": "TEN", "WAS": "WAS",
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


def _section_from_heading(text):
    lowered = _clean(text).lower()
    if lowered.startswith("offense"):
        return "OFFENSE"
    if lowered.startswith("defense"):
        return "DEFENSE"
    if lowered.startswith("special teams"):
        return "SPECIAL TEAMS"
    return ""


def _load_current_depth_chart(team):
    source_code = OURLADS_TEAM_CODES.get(team)
    if not source_code:
        raise RuntimeError(f"No depth-chart source code configured for {team}.")

    url = f"https://www.ourlads.com/nfldepthcharts/depthchart/{source_code}"
    html = _http_get(url)
    soup = BeautifulSoup(html, "html.parser")

    rows = []
    source_slot_order = 0

    # Individual team pages have one clean table per unit. This is much more
    # stable than the all-teams table and keeps Player 1-5 aligned correctly.
    for heading in soup.find_all(["h2", "h3"]):
        section = _section_from_heading(" ".join(heading.stripped_strings))
        if not section:
            continue

        table = heading.find_next("table")
        if table is None:
            continue

        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"], recursive=False)
            if len(cells) < 3:
                continue

            values = [" ".join(cell.stripped_strings).strip() for cell in cells]
            position = _normalize_position(values[0])

            if not position or position in {"POS", "OFF", "DEF", "ST"}:
                continue

            source_slot_order += 1
            slot_id = f"{section}:{source_slot_order}:{position}"

            # Individual page layout:
            # Pos | No. | Player 1 | No | Player 2 | ... | Player 5
            player_slots = [
                (1, 1, 2),
                (2, 3, 4),
                (3, 5, 6),
                (4, 7, 8),
                (5, 9, 10),
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
                        "section": section,
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
                        "section": section,
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
        raise RuntimeError(f"No current depth-chart rows parsed for {team}.")

    return (
        frame.drop_duplicates(
            ["team", "slot_id", "rank", "player_name"],
            keep="first",
        )
        .sort_values(["section", "slot_order", "rank"])
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


def _name_tokens(value):
    text = re.sub(r"[^A-Za-z0-9' -]+", " ", _clean(value))
    return [part for part in text.split() if part]


def _surname_key(value):
    tokens = [
        token.lower().strip(".'-")
        for token in _name_tokens(value)
        if token.lower().strip(".'-") not in {"jr", "sr", "ii", "iii", "iv", "v"}
    ]
    return tokens[-1] if tokens else ""


def _enrich_full_names(frame, season, team):
    """Replace shortened depth-chart names with canonical current-roster names."""
    if frame is None or frame.empty:
        return frame

    try:
        roster = _to_pandas(nfl.load_rosters([int(season)]))
    except Exception:
        return frame

    if roster.empty:
        return frame

    team_col = next(
        (c for c in ["team", "club_code", "club"] if c in roster.columns),
        None,
    )
    name_col = next(
        (c for c in ["full_name", "player_name", "name"] if c in roster.columns),
        None,
    )
    number_col = next(
        (c for c in ["jersey_number", "jersey", "number"] if c in roster.columns),
        None,
    )

    if not team_col or not name_col or not number_col:
        return frame

    roster = roster.copy()
    roster["_team"] = roster[team_col].map(_norm_team)
    roster["_name"] = roster[name_col].map(_clean)
    roster["_number"] = roster[number_col].map(_clean)
    roster = roster[
        (roster["_team"] == team)
        & roster["_name"].ne("")
        & roster["_number"].ne("")
    ].copy()

    if roster.empty:
        return frame

    by_number = {}
    for row in roster.itertuples():
        by_number.setdefault(row._number, []).append(row._name)

    out = frame.copy()

    for index, row in out.iterrows():
        current_name = _clean(row.get("player_name"))
        number = _clean(row.get("number"))

        if not current_name or current_name == "TBD" or not number:
            continue

        candidates = list(dict.fromkeys(by_number.get(number, [])))
        if not candidates:
            continue

        current_surname = _surname_key(current_name)
        surname_matches = [
            candidate
            for candidate in candidates
            if _surname_key(candidate) == current_surname
        ]

        replacement = ""
        if len(surname_matches) == 1:
            replacement = surname_matches[0]
        elif len(candidates) == 1 and len(_name_tokens(current_name)) <= 1:
            # If the parsed source only gave a surname, jersey number is a
            # strong enough tiebreaker when that number is unique on the team.
            replacement = candidates[0]

        if replacement and len(_name_tokens(replacement)) >= 2:
            out.at[index, "player_name"] = replacement

    return out


def load_team_depth_chart(season: int, team: str) -> pd.DataFrame:
    team = _norm_team(team)

    try:
        current = _load_current_depth_chart(team)
        if not current.empty:
            current = _enrich_full_names(current, season, team)
            current.attrs["source"] = "Current depth chart"
            return current
    except Exception as exc:
        current_error = str(exc)
    else:
        current_error = ""

    try:
        fallback = _load_nflverse_fallback(season, team)
    except Exception as exc:
        fallback = pd.DataFrame()
        fallback_error = str(exc)
    else:
        fallback_error = ""

    if fallback.empty:
        empty = pd.DataFrame(
            columns=[
                "team", "section", "position", "slot_id", "slot_order",
                "rank", "player_name", "number", "updated", "source",
            ]
        )
        empty.attrs["live_source_error"] = current_error
        empty.attrs["fallback_error"] = fallback_error
        return empty

    fallback = _enrich_full_names(fallback, season, team)
    fallback.attrs["source"] = "nflverse fallback"
    fallback.attrs["live_source_error"] = current_error
    return fallback


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


def _ensure_slot_metadata(rows):
    """Normalize old/cached depth-chart frames to the current renderer schema."""
    if rows is None or rows.empty:
        return rows

    out = rows.copy()

    if "position" not in out.columns:
        out["position"] = ""
    out["position"] = out["position"].map(_normalize_position)

    if "rank" not in out.columns:
        out["rank"] = 99
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(99)

    if "section" not in out.columns:
        out["section"] = out["position"].map(_position_group)
    else:
        missing_section = out["section"].isna() | out["section"].astype(str).str.strip().eq("")
        out.loc[missing_section, "section"] = out.loc[
            missing_section, "position"
        ].map(_position_group)

    # Older cached frames did not have source slot IDs. Rebuild a stable
    # position slot so the renderer can still work instead of raising KeyError.
    if "slot_order" not in out.columns:
        slot_orders = {}
        rebuilt = []
        for row in out.itertuples():
            section = _clean(getattr(row, "section", "")) or _position_group(
                getattr(row, "position", "")
            )
            position = _normalize_position(getattr(row, "position", ""))
            key = (section, position)

            # All players with the same position in an old frame belong to one
            # legacy slot. Fresh data will preserve duplicate source rows.
            if key not in slot_orders:
                section_count = sum(
                    1 for existing in slot_orders if existing[0] == section
                ) + 1
                slot_orders[key] = section_count

            rebuilt.append(slot_orders[key])
        out["slot_order"] = rebuilt
    else:
        out["slot_order"] = pd.to_numeric(
            out["slot_order"], errors="coerce"
        ).fillna(999).astype(int)

    if "slot_id" not in out.columns:
        out["slot_id"] = (
            out["section"].astype(str)
            + ":"
            + out["slot_order"].astype(str)
            + ":"
            + out["position"].astype(str)
        )
    else:
        missing_slot = out["slot_id"].isna() | out["slot_id"].astype(str).str.strip().eq("")
        out.loc[missing_slot, "slot_id"] = (
            out.loc[missing_slot, "section"].astype(str)
            + ":"
            + out.loc[missing_slot, "slot_order"].astype(str)
            + ":"
            + out.loc[missing_slot, "position"].astype(str)
        )

    for column, default in {
        "player_name": "",
        "number": "",
        "team": "",
    }.items():
        if column not in out.columns:
            out[column] = default

    return out


def _ordered_slots(rows, group):
    rows = _ensure_slot_metadata(rows)
    if rows is None or rows.empty:
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
    if depth_data is None or depth_data.empty:
        return ""

    rows = _ensure_slot_metadata(depth_data)
    if rows is None or rows.empty or "team" not in rows.columns:
        return ""

    rows = rows[rows["team"].map(_norm_team) == _norm_team(team)].copy()
    if rows.empty:
        return ""

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
