from __future__ import annotations

from html import escape

import pandas as pd
import nflreadpy as nfl


TEAM_ALIASES = {
    "GNB": "GB", "KAN": "KC", "LAR": "LA", "NWE": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "LVR": "LV",
    "OAK": "LV", "SDG": "LAC", "STL": "LA", "WSH": "WAS",
}


OFFENSE_LAYOUT = [
    ("WR-L", 8, 17), ("WR-R", 86, 17), ("TE", 83, 38),
    ("LT", 24, 52), ("LG", 37, 52), ("C", 50, 52),
    ("RG", 63, 52), ("RT", 76, 52),
    ("QB", 50, 69), ("RB", 42, 86), ("FB", 59, 86),
]

DEFENSE_LAYOUT = [
    ("FS", 35, 13), ("SS", 65, 13),
    ("CB-L", 8, 73), ("CB-R", 91, 73),
    ("OLB-L", 27, 39), ("MLB", 50, 39), ("OLB-R", 73, 39),
    ("DE-L", 24, 61), ("DT-L", 42, 61), ("DT-R", 58, 61), ("DE-R", 76, 61),
]

SPECIAL_LAYOUT = [
    ("K", 18, 28), ("P", 39, 28), ("LS", 61, 28), ("H", 82, 28),
    ("KR", 27, 70), ("PR", 50, 70), ("KOS", 73, 70),
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
    code = str(value).upper().strip()
    return TEAM_ALIASES.get(code, code)


def _clean(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _first_column(frame, names):
    return next((c for c in names if c in frame.columns), None)


def load_depth_chart_data(season: int) -> pd.DataFrame:
    try:
        frame = _to_pandas(nfl.load_depth_charts([int(season)]))
    except Exception:
        return pd.DataFrame()

    if frame.empty:
        return frame

    team_col = _first_column(frame, ["team", "club_code", "club"])
    name_col = _first_column(frame, ["player_name", "full_name", "name"])
    pos_col = _first_column(frame, ["pos_abb", "position", "pos"])
    rank_col = _first_column(frame, ["pos_rank", "depth", "depth_rank"])
    number_col = _first_column(frame, ["jersey_number", "jersey", "number"])
    espn_col = _first_column(frame, ["espn_id", "espn_player_id"])
    date_col = _first_column(frame, ["dt", "date", "updated"])

    if team_col is None or name_col is None or pos_col is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "team": frame[team_col].map(_norm_team),
        "player_name": frame[name_col].map(_clean),
        "position": frame[pos_col].map(_clean).str.upper(),
    })

    out["rank"] = (
        pd.to_numeric(frame[rank_col], errors="coerce")
        if rank_col
        else 99
    )
    out["number"] = frame[number_col].map(_clean) if number_col else ""
    out["espn_id"] = frame[espn_col].map(_clean) if espn_col else ""

    if date_col:
        out["updated"] = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    else:
        out["updated"] = pd.NaT

    out = out[out["player_name"].ne("") & out["team"].notna()].copy()

    if out["updated"].notna().any():
        latest = out.groupby("team")["updated"].transform("max")
        out = out[(out["updated"].eq(latest)) | out["updated"].isna()].copy()

    out["rank"] = pd.to_numeric(out["rank"], errors="coerce").fillna(99)
    out = out.sort_values(["team", "position", "rank", "player_name"])
    return out.drop_duplicates(["team", "position", "player_name"]).reset_index(drop=True)


def _position_candidates(position):
    p = str(position).upper()
    aliases = {
        "WR": {"WR", "LWR", "RWR", "SWR", "SLWR", "SRWR"},
        "TE": {"TE"},
        "LT": {"LT"},
        "LG": {"LG"},
        "C": {"C"},
        "RG": {"RG"},
        "RT": {"RT"},
        "OL": {"OL", "T", "OT", "G", "OG"},
        "QB": {"QB"},
        "RB": {"RB", "HB"},
        "FB": {"FB"},
        "DE": {"DE", "EDGE", "LDE", "RDE"},
        "DT": {"DT", "NT", "DL", "LDT", "RDT"},
        "OLB": {"OLB", "LOLB", "ROLB", "EDGE"},
        "MLB": {"MLB", "ILB", "LB"},
        "LB": {"LB", "ILB", "MLB", "OLB"},
        "CB": {"CB", "LCB", "RCB", "NB"},
        "FS": {"FS"},
        "SS": {"SS"},
        "S": {"S", "FS", "SS"},
        "K": {"K", "PK"},
        "P": {"P"},
        "LS": {"LS"},
        "H": {"H", "HLD"},
        "KR": {"KR", "KOR"},
        "PR": {"PR"},
        "KOS": {"KOS", "KO", "K"},
    }
    return aliases.get(p, {p})


def _pool(team_rows, group):
    positions = _position_candidates(group)
    rows = team_rows[team_rows["position"].isin(positions)].copy()
    return rows.sort_values(["rank", "player_name"])


def _take(rows, used, count=1):
    picked = []
    for row in rows.itertuples():
        key = str(row.player_name).lower()
        if key in used:
            continue
        picked.append(row)
        used.add(key)
        if len(picked) >= count:
            break
    return picked


def _slot_payload(slot, team_rows, used):
    if slot == "WR-L":
        rows = _pool(team_rows, "WR")
    elif slot == "WR-R":
        rows = _pool(team_rows, "WR")
    elif slot == "CB-L":
        rows = _pool(team_rows, "CB")
    elif slot == "CB-R":
        rows = _pool(team_rows, "CB")
    elif slot.startswith("OLB"):
        rows = _pool(team_rows, "OLB")
        if rows.empty:
            rows = _pool(team_rows, "LB")
    elif slot == "MLB":
        rows = _pool(team_rows, "MLB")
        if rows.empty:
            rows = _pool(team_rows, "LB")
    elif slot.startswith("DE"):
        rows = _pool(team_rows, "DE")
    elif slot.startswith("DT"):
        rows = _pool(team_rows, "DT")
    else:
        rows = _pool(team_rows, slot)

    # Some feeds only expose generic OL/DL/S labels. Use them only when the
    # more specific position is unavailable.
    if rows.empty and slot in {"LT", "LG", "C", "RG", "RT"}:
        rows = _pool(team_rows, "OL")
    if rows.empty and slot in {"FS", "SS"}:
        rows = _pool(team_rows, "S")
    if rows.empty and slot == "FB":
        rows = _pool(team_rows, "RB")

    starters = _take(rows, used, 1)
    starter = starters[0] if starters else None

    # Backup is allowed to repeat a player already used elsewhere only if the
    # source does not expose enough unique names for the slot.
    backup = None
    for row in rows.itertuples():
        if starter is not None and row.player_name == starter.player_name:
            continue
        backup = row
        break

    return starter, backup


def _headshot_url(player) -> str:
    if player is None:
        return ""
    espn_id = _clean(getattr(player, "espn_id", ""))
    if not espn_id:
        return ""
    return f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"


def _card_html(slot, starter, backup, x, y):
    display_slot = slot.replace("-L", "").replace("-R", "")

    if starter is None:
        starter_name = "TBD"
        starter_num = ""
        headshot = ""
    else:
        starter_name = escape(_clean(getattr(starter, "player_name", "")) or "TBD")
        starter_num = escape(_clean(getattr(starter, "number", "")))
        headshot = _headshot_url(starter)

    backup_name = (
        escape(_clean(getattr(backup, "player_name", "")))
        if backup is not None
        else "—"
    )

    number_html = (
        f"<span class='dc-number'>#{starter_num}</span>"
        if starter_num
        else ""
    )
    face_html = (
        f"<img class='dc-face' src='{escape(headshot)}' alt='' loading='lazy' "
        f"onerror=\"this.style.display='none'\" />"
        if headshot
        else "<div class='dc-face dc-face-fallback'></div>"
    )

    # Keep this as one compact HTML string. Streamlit's Markdown parser can
    # otherwise treat indented nested tags as code blocks.
    return (
        f"<div class='dc-player' style='left:{x}%;top:{y}%;'>"
        f"<div class='dc-pos'>{escape(display_slot)} {number_html}</div>"
        f"{face_html}"
        f"<div class='dc-name'>{starter_name}</div>"
        f"<div class='dc-backup'>{backup_name}</div>"
        f"</div>"
    )


def _field_html(title, layout, team_rows):
    used = set()
    cards = []
    for slot, x, y in layout:
        starter, backup = _slot_payload(slot, team_rows, used)
        cards.append(_card_html(slot, starter, backup, x, y))

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
    rows = depth_data[depth_data["team"] == team].copy() if not depth_data.empty else pd.DataFrame()
    if rows.empty:
        return ""

    offense = _field_html("OFFENSE", OFFENSE_LAYOUT, rows)
    defense = _field_html("DEFENSE", DEFENSE_LAYOUT, rows)
    special = _field_html("SPECIAL TEAMS", SPECIAL_LAYOUT, rows)

    return f"""
    <style>
      .dc-wrap {{
        --blue:#087cff;
        --cyan:#36c8ff;
        --panel:#06111f;
        --card:#0a1728;
        --muted:#8fa4be;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
      }}
      .dc-grid {{display:grid;grid-template-columns:1fr;gap:18px}}
      .dc-panel {{
        background:linear-gradient(180deg,#071522,#030914);
        border:1px solid #22354c;border-radius:18px;padding:14px;
        box-shadow:inset 0 0 0 1px rgba(8,124,255,.12);
      }}
      .dc-panel-title {{
        color:white;font-weight:900;letter-spacing:.14em;font-size:18px;
        border-left:4px solid var(--blue);padding-left:12px;margin:2px 0 12px 4px;
      }}
      .dc-field {{
        position:relative;height:430px;overflow:hidden;border-radius:14px;
        background:
          linear-gradient(rgba(255,255,255,.035) 1px,transparent 1px),
          linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px),
          radial-gradient(circle at 50% 50%,rgba(8,124,255,.10),transparent 48%),
          #07131b;
        background-size:100% 52px,70px 100%,100% 100%,100% 100%;
        border:1px solid #24384a;
      }}
      .dc-field:before,.dc-field:after {{
        content:"";position:absolute;left:5%;right:5%;height:2px;background:rgba(255,255,255,.12)
      }}
      .dc-field:before {{top:23%}} .dc-field:after {{top:76%}}
      .dc-midline {{position:absolute;left:3%;right:3%;top:52%;height:2px;background:#087cff;box-shadow:0 0 14px #087cff}}
      .dc-yard {{position:absolute;top:0;bottom:0;width:1px;background:rgba(255,255,255,.06)}}
      .dc-yard.y1{{left:20%}} .dc-yard.y2{{left:40%}} .dc-yard.y3{{left:60%}} .dc-yard.y4{{left:80%}}
      .dc-player {{
        position:absolute;transform:translate(-50%,-50%);
        width:92px;min-height:64px;padding:5px 6px;border-radius:9px;
        background:linear-gradient(180deg,#0d1d31,#07101c);
        border:1px solid #4a6078;box-shadow:0 5px 18px rgba(0,0,0,.35), inset 0 0 0 1px rgba(20,149,255,.15);
        text-align:center;color:#fff;
      }}
      .dc-player:hover {{border-color:var(--cyan);box-shadow:0 0 20px rgba(8,124,255,.28)}}
      .dc-pos {{font-size:10px;font-weight:900;color:#cfe8ff;letter-spacing:.06em;line-height:1}}
      .dc-number {{font-size:8px;color:#72bfff;margin-left:2px}}
      .dc-face {{width:28px;height:28px;border-radius:50%;object-fit:cover;object-position:center 15%;display:block;margin:3px auto 2px;background:#0f2238;border:1px solid rgba(54,200,255,.45)}}
      .dc-face-fallback {{background:radial-gradient(circle at 50% 38%,#52708d 0 25%,#1a3149 26% 55%,#0d1b2a 56%)}}
      .dc-name {{font-size:9px;font-weight:850;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.15}}
      .dc-backup {{font-size:7px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px;border-top:1px solid rgba(255,255,255,.08);padding-top:2px;line-height:1.1}}
      @media (min-width:1050px) {{
        .dc-grid {{grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}
        .dc-panel {{padding:10px;border-radius:14px}}
        .dc-panel-title {{font-size:15px;margin-bottom:8px}}
        .dc-field {{height:430px}}
        .dc-player {{width:72px;min-height:58px;padding:4px 4px}}
        .dc-face {{width:24px;height:24px;margin:2px auto 1px}}
        .dc-name {{font-size:8px}}
        .dc-backup {{font-size:6.5px}}
      }}
      @media (min-width:1450px) {{
        .dc-player {{width:82px}}
        .dc-face {{width:27px;height:27px}}
        .dc-name {{font-size:9px}}
        .dc-backup {{font-size:7px}}
      }}
    </style>
    <div class="dc-wrap"><div class="dc-grid">{offense}{defense}{special}</div></div>
    """


def latest_update_label(depth_data: pd.DataFrame, team: str) -> str:
    if depth_data.empty or "updated" not in depth_data.columns:
        return ""
    rows = depth_data[depth_data["team"] == team]
    values = rows["updated"].dropna()
    if values.empty:
        return ""
    dt = values.max()
    try:
        return dt.strftime("%b %d, %Y")
    except Exception:
        return ""
