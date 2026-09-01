import json

import streamlit as st
import pandas as pd

from access_control import require_access, render_access_sidebar

from model import (
    FEATURES,
    DISPLAY_NAMES,
    available_teams,
    load_nfl_data,
    predict_matchup,
    train_model,
    upcoming_games,
)
from offseason import OFFSEASON_FEATURES, OFFSEASON_DISPLAY_NAMES
from injuries import injury_status_counts, latest_team_injuries, load_injury_data
from team_data import (
    current_season_summary,
    recent_team_games,
    team_record,
    team_upcoming_games,
)

MODEL_CACHE_VERSION = "player-impact-role-v2"


TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LA": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF": "San Francisco 49ers",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
}



TEAM_HOME_LOCATIONS = {
    "ARI": "Glendale, AZ", "ATL": "Atlanta, GA", "BAL": "Baltimore, MD",
    "BUF": "Orchard Park, NY", "CAR": "Charlotte, NC", "CHI": "Chicago, IL",
    "CIN": "Cincinnati, OH", "CLE": "Cleveland, OH", "DAL": "Arlington, TX",
    "DEN": "Denver, CO", "DET": "Detroit, MI", "GB": "Green Bay, WI",
    "HOU": "Houston, TX", "IND": "Indianapolis, IN", "JAX": "Jacksonville, FL",
    "KC": "Kansas City, MO", "LV": "Las Vegas, NV", "LAC": "Inglewood, CA",
    "LA": "Inglewood, CA", "MIA": "Miami Gardens, FL", "MIN": "Minneapolis, MN",
    "NE": "Foxborough, MA", "NO": "New Orleans, LA", "NYG": "East Rutherford, NJ",
    "NYJ": "East Rutherford, NJ", "PHI": "Philadelphia, PA", "PIT": "Pittsburgh, PA",
    "SEA": "Seattle, WA", "SF": "Santa Clara, CA", "TB": "Tampa, FL",
    "TEN": "Nashville, TN", "WAS": "Landover, MD",
}

POSITION_GROUP_ORDER = ["All", "QB", "RB", "WR", "TE", "OL", "DL", "LB", "DB", "ST", "Other"]

TEAM_LOGO_CODES = {
    "ARI": "ari",
    "ATL": "atl",
    "BAL": "bal",
    "BUF": "buf",
    "CAR": "car",
    "CHI": "chi",
    "CIN": "cin",
    "CLE": "cle",
    "DAL": "dal",
    "DEN": "den",
    "DET": "det",
    "GB": "gb",
    "HOU": "hou",
    "IND": "ind",
    "JAX": "jax",
    "KC": "kc",
    "LV": "lv",
    "LAC": "lac",
    "LA": "lar",
    "MIA": "mia",
    "MIN": "min",
    "NE": "ne",
    "NO": "no",
    "NYG": "nyg",
    "NYJ": "nyj",
    "PHI": "phi",
    "PIT": "pit",
    "SEA": "sea",
    "SF": "sf",
    "TB": "tb",
    "TEN": "ten",
    "WAS": "wsh",
}


def team_name(code):
    return TEAM_NAMES.get(code, code)


def team_label(code):
    return f"{team_name(code)} ({code})"


def team_logo_url(code):
    logo_code = TEAM_LOGO_CODES.get(code, str(code).lower())
    return f"https://a.espncdn.com/i/teamlogos/nfl/500/{logo_code}.png"


def show_team_logo(slot, code):
    slot.image(team_logo_url(code), width=125)


def split_player_names(value):
    if pd.isna(value) or not str(value).strip():
        return []
    return [name.strip() for name in str(value).split("|") if name.strip()]


def show_player_list(title, value, empty_text):
    st.markdown(f"#### {title}")
    names = split_player_names(value)
    if not names:
        st.caption(empty_text)
        return
    st.markdown("\n".join(f"- {name}" for name in names))


def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def format_kickoff(game):
    gameday = pd.to_datetime(game.get("gameday"), errors="coerce")
    date_text = "Date TBD"
    if pd.notna(gameday):
        date_text = f"{gameday.strftime('%b')} {gameday.day}, {gameday.year}"

    time_text = "TBD"
    raw_time = clean_text(game.get("gametime"))
    if raw_time:
        try:
            parsed = pd.to_datetime(raw_time, format="%H:%M", errors="raise")
            time_text = parsed.strftime("%I:%M %p").lstrip("0")
        except Exception:
            time_text = raw_time

    return f"{time_text} ET • {date_text}"


def format_game_location(game):
    stadium = clean_text(game.get("stadium"))
    location_flag = clean_text(game.get("location")).lower()
    home_team = clean_text(game.get("home_team"))

    if location_flag == "neutral":
        return f"{stadium} • Neutral site" if stadium else "Neutral site"

    city = TEAM_HOME_LOCATIONS.get(home_team, "")
    if stadium and city:
        return f"{stadium} • {city}"
    return stadium or city or "Location TBD"


def parse_player_details(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def movement_dataframe(players, position_group):
    filtered = [
        player for player in players
        if position_group == "All" or player.get("position_group") == position_group
    ]
    filtered = sorted(
        filtered,
        key=lambda player: (-float(player.get("impact_score", 0.0)), player.get("name", "")),
    )

    rows = []
    for rank, player in enumerate(filtered, start=1):
        rows.append({
            "Rank": rank,
            "Player": player.get("name", "Unknown"),
            "Position": player.get("position", "—"),
            "Projected role": player.get("projected_role", "Role unavailable"),
            "Recent usage": f"{float(player.get('recent_snap_share', 0.0)):.0%}",
            "Production grade": f"{100.0 * float(player.get('recent_production', 0.0)):.0f}/100",
            "Impact score": round(float(player.get("impact_score", 0.0)), 1),
        })
    return pd.DataFrame(rows)


def render_movement_column(title, players, position_group, empty_text):
    st.markdown(f"### {title}")
    table = movement_dataframe(players, position_group)

    if table.empty:
        st.caption(empty_text)
        return

    top = table.iloc[0]
    st.markdown(
        f"**Top move:** {top['Player']} · {top['Position']} · {top['Projected role']}  "
        f"<span class='small-note'>Projected impact {top['Impact score']:.1f}/100</span>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        height=min(420, 78 + (len(table) * 35)),
    )


def record_label(team_games, team, season):
    record = team_record(team_games, team, season)
    return f"{record['wins']}-{record['losses']}"


def recent_form_label(team_games, team, limit=5):
    recent = recent_team_games(team_games, team, limit)
    if recent.empty:
        return "No recent games"
    return " ".join(recent["result"].astype(str).tolist())


def injury_table_for_team(injury_data, team):
    latest = latest_team_injuries(injury_data, team)
    if latest.empty:
        return latest

    display = latest.copy()
    rename_map = {
        "full_name": "Player",
        "position": "Pos",
        "report_primary_injury": "Injury",
        "report_secondary_injury": "Secondary",
        "report_status": "Game status",
        "practice_status": "Practice",
        "date_modified": "Updated",
        "week": "Week",
    }
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})

    if "Updated" in display.columns:
        display["Updated"] = pd.to_datetime(display["Updated"], errors="coerce").dt.strftime("%b %d, %I:%M %p")

    priority = {"out": 0, "doubtful": 1, "questionable": 2}
    if "Game status" in display.columns:
        display["_priority"] = (
            display["Game status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .map(priority)
            .fillna(9)
        )
        display = display.sort_values(["_priority", "Pos", "Player"]).drop(columns=["_priority"])

    return display.reset_index(drop=True)


def render_injury_summary(injury_data, team, compact=False):
    latest = latest_team_injuries(injury_data, team)
    counts = injury_status_counts(latest)

    if compact:
        st.caption(
            f"Out: {counts['Out']} • Doubtful: {counts['Doubtful']} • "
            f"Questionable: {counts['Questionable']}"
        )
        return

    i1, i2, i3 = st.columns(3)
    i1.metric("Out", counts["Out"])
    i2.metric("Doubtful", counts["Doubtful"])
    i3.metric("Questionable", counts["Questionable"])

    table = injury_table_for_team(injury_data, team)
    if table.empty:
        st.info("No official injury report is available for this team yet.")
    else:
        st.dataframe(table, hide_index=True, use_container_width=True)


def render_team_recent_games(team_games, team):
    recent = recent_team_games(team_games, team, 5)
    if recent.empty:
        st.info("No completed games found.")
        return

    display = recent.copy()
    display["Date"] = pd.to_datetime(display["gameday"], errors="coerce").dt.strftime("%b %d, %Y")
    display["Opponent"] = display["opponent"].map(lambda code: team_name(code))
    display["Result"] = display["result"]
    display["Score"] = display["score"]
    display["Point diff"] = display["point_diff_display"]
    st.dataframe(
        display[["Date", "Opponent", "Result", "Score", "Point diff"]],
        hide_index=True,
        use_container_width=True,
    )

    chart = display[["Opponent", "point_diff_display"]].copy()
    chart = chart.rename(columns={"point_diff_display": "Point differential"}).set_index("Opponent")
    st.bar_chart(chart)


def render_team_upcoming(team, schedules, limit=5):
    future = team_upcoming_games(schedules, team, limit)
    if future.empty:
        st.info("No upcoming games found.")
        return

    for _, game in future.iterrows():
        home = clean_text(game.get("home_team"))
        away = clean_text(game.get("away_team"))
        opponent = away if home == team else home
        venue = format_game_location(game)
        st.markdown(
            f"**{team_name(opponent)}** — {format_kickoff(game)}  " + "\n" + f"📍 {venue}"
        )


st.set_page_config(
    page_title="NFL Prediction Lab",
    page_icon="🏈",
    layout="wide",
)

current_user = require_access()
render_access_sidebar(current_user)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      .metric-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        padding: 1rem 1.2rem;
      }
      .small-note {opacity: .75; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏈 NFL Prediction Lab")
st.caption(
    "Educational sports-analytics dashboard. It estimates game outcomes from historical "
    "team performance plus offseason roster and draft changes."
)


@st.cache_data(ttl=900, show_spinner=False)
def get_data():
    return load_nfl_data([2023, 2024, 2025, 2026])


@st.cache_resource(show_spinner=False)
def get_model(_schedules, _team_stats, data_signature, model_cache_version):
    return train_model(_schedules, _team_stats)


@st.cache_data(ttl=900, show_spinner=False)
def get_injuries(season):
    return load_injury_data(season)


with st.sidebar:
    st.header("Data controls")
    if st.button("↻ Refresh NFL data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown(
        """
        **How updating works**

        The data cache expires every 15 minutes while the app is being used.
        New final scores are incorporated when nflverse publishes them.

        The V2 model also rebuilds offseason features from current roster,
        draft, player-stat, and trade datasets.
        """
    )

try:
    with st.spinner("Loading NFL data, offseason moves, and training model..."):
        schedules, team_stats = get_data()

        completed_mask = (
            pd.to_numeric(schedules["home_score"], errors="coerce").notna()
            & pd.to_numeric(schedules["away_score"], errors="coerce").notna()
        )
        signature = (
            int(completed_mask.sum()),
            str(schedules.loc[completed_mask, "game_id"].iloc[-1])
            if completed_mask.any()
            else "none",
        )
        bundle = get_model(
            schedules,
            team_stats,
            signature,
            MODEL_CACHE_VERSION,
        )
        current_season = int(
            pd.to_numeric(schedules["season"], errors="coerce").dropna().max()
        )
        injuries = get_injuries(current_season)

except Exception as exc:
    st.error("The dashboard could not load or train the NFL model.")
    st.exception(exc)
    st.stop()


teams = available_teams(bundle)

tab_predict, tab_team, tab_upcoming, tab_offseason, tab_injuries, tab_accuracy, tab_method = st.tabs(
    [
        "Matchup Predictor",
        "Team Hub",
        "Upcoming Games",
        "Offseason Changes",
        "Player Status",
        "Model Accuracy",
        "How It Works",
    ]
)

with tab_predict:
    st.subheader("Compare two teams")

    c1, c2 = st.columns(2)
    default_away = teams.index("BUF") if "BUF" in teams else 0
    default_home = teams.index("KC") if "KC" in teams else min(1, len(teams) - 1)

    with c1:
        away_logo = st.empty()
        away = st.selectbox(
            "Away team",
            teams,
            index=default_away,
            format_func=team_label,
        )
        show_team_logo(away_logo, away)

    with c2:
        home_logo = st.empty()
        home = st.selectbox(
            "Home team",
            teams,
            index=default_home,
            format_func=team_label,
        )
        show_team_logo(home_logo, home)

    if away == home:
        st.warning("Choose two different teams.")
    else:
        result = predict_matchup(bundle, away, home)

        away_name = team_name(away)
        home_name = team_name(home)
        predicted_name = team_name(result["predicted_winner"])

        with st.container(border=True):
            away_card, center_card, home_card = st.columns([1, 1.25, 1])

            with away_card:
                st.image(team_logo_url(away), width=105)
                st.markdown(f"### {away_name}")
                st.caption(
                    f"Away • Record {record_label(bundle.team_games, away, current_season)} • "
                    f"Last 5: {recent_form_label(bundle.team_games, away)}"
                )

            with center_card:
                st.markdown("<div style='text-align:center'><b>MODEL MATCHUP</b></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div style='text-align:center;font-size:1.35rem;margin:.5rem 0'>"
                    f"<b>{away_name}</b> @ <b>{home_name}</b></div>",
                    unsafe_allow_html=True,
                )
                st.metric("Model pick", predicted_name)
                p_away, p_home = st.columns(2)
                p_away.metric(away, f"{result['away_probability']:.1%}")
                p_home.metric(home, f"{result['home_probability']:.1%}")

            with home_card:
                st.image(team_logo_url(home), width=105)
                st.markdown(f"### {home_name}")
                st.caption(
                    f"Home • Record {record_label(bundle.team_games, home, current_season)} • "
                    f"Last 5: {recent_form_label(bundle.team_games, home)}"
                )

        st.progress(result["confidence"])
        st.caption(
            f"Model confidence in its higher-probability side: {result['confidence']:.1%}. "
            "A probability is not a guarantee."
        )

        st.markdown("#### Player availability")
        inj_away, inj_home = st.columns(2)
        with inj_away:
            st.markdown(f"**{away_name}**")
            render_injury_summary(injuries, away, compact=True)
        with inj_home:
            st.markdown(f"**{home_name}**")
            render_injury_summary(injuries, home, compact=True)

        st.markdown("#### Biggest statistical drivers")
        factor_df = pd.DataFrame(result["factors"][:8])
        factor_df = factor_df[["category", "factor", "leans", "raw_difference", "model_contribution"]]
        factor_df["leans"] = factor_df["leans"].map(
            lambda value: team_name(value) if value in TEAM_NAMES else value
        )
        factor_df.columns = [
            "Category",
            "Factor",
            "Leans toward",
            "Home − away difference",
            "Model contribution",
        ]
        st.dataframe(factor_df, hide_index=True, use_container_width=True)

        st.markdown("#### Current team profile")
        profile_rows = []
        for f in FEATURES:
            profile_rows.append(
                {
                    "Metric": DISPLAY_NAMES[f],
                    away_name: result["away_snapshot"][f],
                    home_name: result["home_snapshot"][f],
                }
            )
        st.dataframe(pd.DataFrame(profile_rows), hide_index=True, use_container_width=True)


with tab_team:
    st.subheader("Team Hub")
    selected_team = st.selectbox(
        "Choose a team",
        teams,
        format_func=team_label,
        key="team_hub_team",
    )

    hub_left, hub_right = st.columns([1, 3])
    with hub_left:
        st.image(team_logo_url(selected_team), width=150)
    with hub_right:
        st.markdown(f"## {team_name(selected_team)}")
        record = team_record(bundle.team_games, selected_team, current_season)
        summary = current_season_summary(bundle.team_games, selected_team, current_season)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{current_season} Record", f"{record['wins']}-{record['losses']}")
        m2.metric("Avg points", f"{summary['points_for']:.1f}")
        m3.metric("Avg allowed", f"{summary['points_against']:.1f}")
        m4.metric("Avg point diff", f"{summary['point_diff']:+.1f}")

    recent_col, schedule_col = st.columns(2)
    with recent_col:
        st.markdown("### Last 5 completed games")
        render_team_recent_games(bundle.team_games, selected_team)
    with schedule_col:
        st.markdown("### Next games")
        render_team_upcoming(selected_team, bundle.schedules, 5)

    st.markdown("### Latest player status")
    render_injury_summary(injuries, selected_team)

    if not bundle.offseason.empty:
        latest_offseason = bundle.offseason[
            (bundle.offseason["team"] == selected_team)
            & (bundle.offseason["season"] == bundle.offseason["season"].max())
        ]
        if not latest_offseason.empty:
            row = latest_offseason.iloc[-1]
            additions = parse_player_details(row.get("added_player_details", ""))
            departures = parse_player_details(row.get("departed_player_details", ""))

            st.markdown("### Offseason snapshot")
            add_col, loss_col = st.columns(2)
            with add_col:
                top_adds = movement_dataframe(additions, "All").head(5)
                st.markdown("**Top additions**")
                if top_adds.empty:
                    st.caption("No additions found.")
                else:
                    st.dataframe(top_adds, hide_index=True, use_container_width=True)
            with loss_col:
                top_losses = movement_dataframe(departures, "All").head(5)
                st.markdown("**Top departures**")
                if top_losses.empty:
                    st.caption("No departures found.")
                else:
                    st.dataframe(top_losses, hide_index=True, use_container_width=True)


with tab_upcoming:
    st.subheader("Upcoming schedule")
    st.caption("Kickoff times are shown in Eastern Time.")

    upcoming = upcoming_games(bundle.schedules, 300)

    week_options = ["All Weeks"] + [f"Week {week}" for week in range(1, 17)]
    selected_week = st.selectbox(
        "Filter by week",
        week_options,
        index=1,
        key="upcoming_week_filter",
    )

    if selected_week != "All Weeks" and not upcoming.empty:
        week_number = int(selected_week.split()[-1])
        upcoming = upcoming[
            pd.to_numeric(upcoming["week"], errors="coerce") == week_number
        ].copy()

    if upcoming.empty:
        if selected_week == "All Weeks":
            st.info("No upcoming non-preseason games were found in the current schedule file.")
        else:
            st.info(f"No upcoming games were found for {selected_week}.")
    else:
        for _, game in upcoming.iterrows():
            away_team = clean_text(game.get("away_team"))
            home_team = clean_text(game.get("home_team"))
            week = clean_text(game.get("week"))

            with st.container(border=True):
                header_left, header_right = st.columns([1, 2])
                with header_left:
                    st.caption(f"Week {week}" if week else "Upcoming game")
                with header_right:
                    st.markdown(
                        f"<div style='text-align:right;font-weight:600'>{format_kickoff(game)}</div>",
                        unsafe_allow_html=True,
                    )

                away_col, at_col, home_col = st.columns([1, 0.24, 1])
                with away_col:
                    st.image(team_logo_url(away_team), width=72)
                    st.markdown(f"**{team_name(away_team)}**")
                    st.caption("Away")
                with at_col:
                    st.markdown(
                        "<div style='text-align:center;padding-top:38px;font-size:1.35rem;font-weight:700'>@</div>",
                        unsafe_allow_html=True,
                    )
                with home_col:
                    st.image(team_logo_url(home_team), width=72)
                    st.markdown(f"**{team_name(home_team)}**")
                    st.caption("Home")

                st.markdown(f"📍 **{format_game_location(game)}**")

        st.caption(
            "Use any matchup from this list in the Matchup Predictor. "
            "Final scores are incorporated after a data refresh."
        )


with tab_offseason:
    st.subheader("2026 offseason roster, trade, and draft changes")
    st.caption(
        "Select a team and position group to browse offseason additions and departures, "
        "ranked from highest estimated impact to lowest."
    )

    if bundle.offseason.empty:
        st.warning("Offseason data is not currently available from the source.")
    else:
        latest_season = int(bundle.offseason["season"].max())
        latest = bundle.offseason[bundle.offseason["season"] == latest_season].copy()
        offseason_teams = [team for team in teams if team in set(latest["team"].astype(str))]

        if not offseason_teams:
            st.warning("No team-level offseason rows are available for the latest season.")
        else:
            team_left, team_right = st.columns([1, 2])
            with team_left:
                selected_logo = st.empty()
                selected_team = st.selectbox(
                    f"{latest_season} team",
                    offseason_teams,
                    format_func=team_label,
                    key="offseason_team",
                )
                show_team_logo(selected_logo, selected_team)

            team_row = latest[latest["team"] == selected_team].iloc[-1]
            added_details = parse_player_details(team_row.get("added_player_details", ""))
            departed_details = parse_player_details(team_row.get("departed_player_details", ""))

            available_groups = {
                player.get("position_group", "Other")
                for player in added_details + departed_details
            }
            position_options = [
                group for group in POSITION_GROUP_ORDER
                if group == "All" or group in available_groups
            ]

            with team_right:
                st.markdown(f"### {team_name(selected_team)}")
                position_group = st.selectbox(
                    "Position group",
                    position_options,
                    key="offseason_position_group",
                    help="Choose QB, RB, WR, TE, OL, DL, LB, DB, ST, or view everyone.",
                )
                st.caption(
                    "Projected impact estimates first-season contribution from current depth-chart role, "
                    "roster status, recent NFL production, snap share, and draft capital. Current role is "
                    "the strongest multiplier, so backups and practice-squad players are heavily discounted."
                )

            additions_col, losses_col = st.columns(2)
            with additions_col:
                render_movement_column(
                    "⬆️ Acquired / Added",
                    added_details,
                    position_group,
                    "No additions found for this position group.",
                )
            with losses_col:
                render_movement_column(
                    "⬇️ Departed / Lost",
                    departed_details,
                    position_group,
                    "No departures found for this position group.",
                )

            with st.expander("Draft class and trade log"):
                d1, d2, d3 = st.columns(3)
                with d1:
                    show_player_list(
                        f"{latest_season} draft class",
                        team_row.get("drafted_players", ""),
                        "No drafted-player names were found.",
                    )
                with d2:
                    show_player_list(
                        "Trade acquisitions",
                        team_row.get("trade_added_players", ""),
                        "No player trade acquisitions were found.",
                    )
                with d3:
                    show_player_list(
                        "Trade departures",
                        team_row.get("trade_departed_players", ""),
                        "No player trade departures were found.",
                    )

            with st.expander("Model offseason inputs"):
                model_rows = []
                for feature in OFFSEASON_FEATURES:
                    value = team_row.get(feature, 0.0)
                    if feature in {"roster_continuity", "addition_weight", "departure_weight"}:
                        display_value = f"{float(value):.1%}"
                    elif feature == "qb_returning":
                        display_value = "Yes" if float(value) >= 0.5 else "No"
                    else:
                        display_value = f"{float(value):.2f}"
                    model_rows.append(
                        {
                            "Model input": OFFSEASON_DISPLAY_NAMES[feature],
                            "Value": display_value,
                        }
                    )
                st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)

            st.caption(
                "Roster additions and departures come from comparing prior-season and current-season "
                "nflverse rosters. Draft and trade names come from nflverse datasets."
            )


with tab_injuries:
    st.subheader("Player Status")
    st.caption(
        "Latest official weekly NFL injury-report information from nflverse. "
        "This panel is informational and does not change the model prediction."
    )

    injury_team = st.selectbox(
        "Team",
        teams,
        format_func=team_label,
        key="injury_team",
    )

    st.image(team_logo_url(injury_team), width=100)
    st.markdown(f"### {team_name(injury_team)}")
    render_injury_summary(injuries, injury_team)


with tab_accuracy:
    st.subheader("Chronological validation")

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Validation accuracy",
        f"{bundle.validation_accuracy:.1%}" if bundle.validation_accuracy is not None else "N/A",
    )
    m2.metric(
        "Brier score",
        f"{bundle.validation_brier:.3f}" if bundle.validation_brier is not None else "N/A",
    )
    m3.metric("Training games", f"{len(bundle.training_games):,}")

    st.markdown(
        """
        The dashboard sorts games by date, trains a temporary model on the older 80%,
        and tests it on the newest 20%. This is more realistic than randomly mixing
        old and new games.

        **Accuracy** measures how often the higher-probability team won.  
        **Brier score** measures probability quality; lower is better.
        """
    )

    by_season = (
        bundle.training_games.groupby("season")
        .agg(games=("game_id", "count"), home_win_rate=("home_win", "mean"))
        .reset_index()
    )
    st.dataframe(by_season, hide_index=True, use_container_width=True)


with tab_method:
    st.subheader("Model design")
    st.markdown(
        """
        This version uses **logistic regression** and combines two groups of information.

        **Performance features**
        - 8-game point differential
        - 8-game points scored and allowed
        - 8-game offensive yards and yards allowed
        - 8-game turnover margin
        - last-5 win percentage
        - home/away performance
        - opponent strength

        **Offseason features**
        - roster continuity
        - weighted veteran additions
        - weighted veteran departures
        - draft class impact
        - primary-QB continuity
        - player trade additions
        - player trade departures

        Historical game features are constructed using information available before each game.
        The final model is retrained on completed games from 2023 onward, with newer seasons
        receiving slightly more weight.
        """
    )

    st.info(
        "The offseason layer is an estimate. It does not yet know every player's true future "
        "performance, exact starting role, coaching fit, injuries, or preseason development. "
        "Those are candidates for later versions."
    )
