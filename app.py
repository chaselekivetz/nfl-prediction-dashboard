import json
import math

import streamlit as st
import pandas as pd
import pydeck as pdk

from access_control import require_access, render_access_sidebar, render_invite_page

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
from fun_features import (
    challenge_active_week,
    completed_challenge_weeks,
    current_week,
    games_for_week,
    league_map_frame,
    regular_season_weeks,
    upcoming_games_for_week,
    week_is_complete,
)
from prediction_game import (
    POSITION_STATS,
    STAT_LABELS,
    favorite_team,
    game_is_open,
    leaderboard,
    player_display_name,
    profile_username,
    load_skill_roster,
    load_weekly_player_stats,
    save_stat_prediction,
    save_winner_pick,
    save_username,
    set_favorite_team,
    stat_predictions_for_user,
    user_score,
    weekly_user_score,
    winner_pick_for_user,
)
from team_history import team_history
from depth_chart import depth_chart_html, latest_update_label, load_depth_chart_data

MODEL_CACHE_VERSION = "player-impact-role-v4-full-trade-injury-audit"
GRIDIRON_LOGO_PATH = "assets/gridiron_central_logo.svg"
APP_DISPLAY_NAME = "Gridiron Central"


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
        "status_detail": "Status detail",
        "practice_status": "Practice",
        "date_modified": "Updated",
        "source": "Source",
        "week": "Week",
    }
    display = display.rename(columns={k: v for k, v in rename_map.items() if k in display.columns})

    if "Updated" in display.columns:
        display["Updated"] = pd.to_datetime(display["Updated"], errors="coerce").dt.strftime("%b %d, %I:%M %p")

    priority = {"out": 0, "unavailable": 1, "doubtful": 2, "questionable": 3}
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
            f"Questionable: {counts['Questionable']} • "
            f"IR/PUP/NFI: {counts['Unavailable']}"
        )
        return

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Out", counts["Out"])
    i2.metric("Doubtful", counts["Doubtful"])
    i3.metric("Questionable", counts["Questionable"])
    i4.metric("IR / PUP / NFI", counts["Unavailable"])

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


def projected_margin_from_result(result, away, home):
    """Translate win probability into an approximate matchup margin for display."""
    try:
        home_probability = float(result.get("home_probability", 0.5))
    except Exception:
        home_probability = 0.5

    home_probability = min(max(home_probability, 0.05), 0.95)
    signed_home_margin = 8.0 * math.log(home_probability / (1.0 - home_probability))
    signed_home_margin = max(min(signed_home_margin, 21.0), -21.0)

    if abs(signed_home_margin) < 0.5:
        return {
            "team": None,
            "margin": 0.0,
            "label": "Approx. even matchup",
        }

    favored_team = home if signed_home_margin > 0 else away
    margin = abs(signed_home_margin)
    return {
        "team": favored_team,
        "margin": margin,
        "label": f"{team_name(favored_team)} by {margin:.1f}",
    }


def matchup_projection(bundle, away, home):
    cache = st.session_state.setdefault("_matchup_projection_cache", {})
    cache_key = (id(bundle), str(away), str(home))
    if cache_key in cache:
        return cache[cache_key]

    try:
        result = predict_matchup(bundle, away, home)
    except Exception:
        projection = (None, None)
    else:
        projection = (
            result,
            projected_margin_from_result(result, away, home),
        )

    cache[cache_key] = projection
    return projection


def game_of_week(bundle, weekly_games):
    if weekly_games.empty:
        return None, None

    best_game = None
    best_result = None
    best_distance = None

    for _, game in weekly_games.iterrows():
        away = clean_text(game.get("away_team"))
        home = clean_text(game.get("home_team"))
        if not away or not home or away == home:
            continue
        try:
            result = predict_matchup(bundle, away, home)
        except Exception:
            continue

        distance = abs(float(result["home_probability"]) - 0.5)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_game = game
            best_result = result

    return best_game, best_result


def format_game_choice(game):
    away = clean_text(game.get("away_team"))
    home = clean_text(game.get("home_team"))
    return f"{team_name(away)} @ {team_name(home)} — {format_kickoff(game)}"


def render_game_of_week(game, result):
    if game is None or result is None:
        st.info("Game of the Week will appear when upcoming games are available.")
        return

    away = clean_text(game.get("away_team"))
    home = clean_text(game.get("home_team"))

    with st.container(border=True):
        st.markdown("### ⭐ Game of the Week")
        st.caption("Selected as the closest matchup in the current model.")

        away_col, middle_col, home_col = st.columns([1, 1.1, 1])
        with away_col:
            st.image(team_logo_url(away), width=90)
            st.markdown(f"**{team_name(away)}**")
            st.caption(f"Record: {record_label(bundle.team_games, away, current_season)}")
        with middle_col:
            st.markdown(
                f"<div style='text-align:center;font-size:1.25rem;font-weight:700'>"
                f"{result['away_probability']:.1%} &nbsp; vs &nbsp; {result['home_probability']:.1%}"
                f"</div>",
                unsafe_allow_html=True,
            )
            margin = projected_margin_from_result(result, away, home)
            st.markdown(f"**Model projected margin:** {margin['label']}")
            st.caption("Approximate model estimate for matchup comparison; not a market line.")
            st.caption(format_kickoff(game))
            st.caption(format_game_location(game))
        with home_col:
            st.image(team_logo_url(home), width=90)
            st.markdown(f"**{team_name(home)}**")
            st.caption(f"Record: {record_label(bundle.team_games, home, current_season)}")


def completed_winner(game):
    home_score = pd.to_numeric(game.get("home_score"), errors="coerce")
    away_score = pd.to_numeric(game.get("away_score"), errors="coerce")
    if pd.isna(home_score) or pd.isna(away_score) or home_score == away_score:
        return None
    return clean_text(game.get("home_team")) if home_score > away_score else clean_text(game.get("away_team"))


st.set_page_config(
    page_title=APP_DISPLAY_NAME,
    page_icon="🏈",
    layout="wide",
)

current_user = require_access()
sidebar_action = render_access_sidebar(current_user)

if sidebar_action == "invite_friend" and current_user.is_admin:
    if st.button("← Back to dashboard", key="back_from_invite_friend"):
        st.session_state.pop("admin_page", None)
        st.rerun()
    render_invite_page(current_user)
    st.stop()

app_username = profile_username(current_user.email)
if not app_username:
    st.title("🏈 Create your NFL username")
    st.caption(
        "This is the name other users will see on the Home screen and Weekly Challenge leaderboard."
    )
    with st.form("create_username_form"):
        new_username = st.text_input(
            "Username",
            max_chars=20,
            placeholder="Example: ChaseNFL",
            help="3–20 characters. Letters, numbers, and underscores only.",
        )
        create_username = st.form_submit_button(
            "Create username",
            type="primary",
            use_container_width=True,
        )

    if create_username:
        try:
            save_username(current_user.email, current_user.name, new_username)
            st.success("Username created.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.info("Your email stays private and is only used for sign-in and account access.")
    st.stop()

app_username = player_display_name(current_user.email, current_user.name)

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

brand_logo, brand_title = st.columns([0.13, 0.87], vertical_alignment="center")
with brand_logo:
    st.image(GRIDIRON_LOGO_PATH, width=118)
with brand_title:
    st.markdown(f"# {APP_DISPLAY_NAME}")

st.caption(
    "NFL forecasts, weekly prediction challenges, team intelligence, schedules, "
    "depth charts, player status, and league exploration."
)


@st.cache_data(ttl=900, show_spinner=False)
def get_data():
    return load_nfl_data([2023, 2024, 2025, 2026])


@st.cache_resource(show_spinner=False)
def get_model(_schedules, _team_stats, data_signature, model_cache_version):
    return train_model(_schedules, _team_stats)


@st.cache_data(ttl=300, show_spinner=False)
def get_injuries(season):
    return load_injury_data(season)


@st.cache_data(ttl=1800, show_spinner=False)
def get_depth_charts(season):
    return load_depth_chart_data(season)


@st.cache_data(ttl=900, show_spinner=False)
def get_challenge_roster(season):
    return load_skill_roster(season)


@st.cache_data(ttl=900, show_spinner=False)
def get_challenge_stats(season):
    return load_weekly_player_stats(season)


with st.sidebar:
    st.header("Data controls")
    if st.button("↻ Refresh NFL data", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pop("_matchup_projection_cache", None)
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
        depth_charts = get_depth_charts(current_season)
        challenge_roster = get_challenge_roster(current_season)
        challenge_stats = get_challenge_stats(current_season)
        active_week = current_week(schedules, current_season)
        active_challenge_week = challenge_active_week(schedules, current_season)
        challenge_completed_weeks = completed_challenge_weeks(schedules, current_season)
        challenge_regular_weeks = regular_season_weeks(schedules, current_season)
        challenge_season_finished = bool(
            challenge_regular_weeks
            and len(challenge_completed_weeks) == len(challenge_regular_weeks)
        )

except Exception as exc:
    st.error("The dashboard could not load or train the NFL model.")
    st.exception(exc)
    st.stop()


teams = available_teams(bundle)

tab_home, tab_challenge, tab_predict, tab_team, tab_depth, tab_upcoming, tab_offseason, tab_injuries, tab_map, tab_accuracy, tab_method = st.tabs(
    [
        "Home",
        "Weekly Challenge",
        "Matchup Predictor",
        "Team Hub",
        "Depth Chart",
        "Upcoming Games",
        "Offseason Changes",
        "Player Status",
        "NFL Map",
        "Model Accuracy",
        "How It Works",
    ]
)

with tab_home:
    st.subheader(f"Week {active_week} Home")

    stored_favorite = favorite_team(current_user.email)
    default_favorite = stored_favorite if stored_favorite in teams else (teams[0] if teams else None)
    favorite = default_favorite

    if "show_favorite_picker" not in st.session_state:
        st.session_state["show_favorite_picker"] = False

    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.markdown(f"### Welcome, {app_username}")
        st.caption("Your weekly NFL dashboard, prediction challenge, and team explorer.")

        if "show_username_editor" not in st.session_state:
            st.session_state["show_username_editor"] = False

        if st.button(
            "Edit username",
            key="edit_username_button",
        ):
            st.session_state["show_username_editor"] = True

        if st.session_state["show_username_editor"]:
            with st.form("edit_username_form"):
                edited_username = st.text_input(
                    "Username",
                    value=app_username,
                    max_chars=20,
                    help="3–20 characters. Letters, numbers, and underscores only.",
                )
                save_username_change = st.form_submit_button(
                    "Save username",
                    type="primary",
                )

            if save_username_change:
                try:
                    save_username(
                        current_user.email,
                        app_username,
                        edited_username,
                    )
                    st.session_state["show_username_editor"] = False
                    st.success("Username updated.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

    with top_right:
        if teams and favorite:
            st.markdown("**Favorite team**")
            favorite_logo, favorite_name_col = st.columns([1, 2])
            with favorite_logo:
                st.image(team_logo_url(favorite), width=64)
            with favorite_name_col:
                st.markdown(f"**{team_name(favorite)}**")

            if st.button(
                "Change favorite team",
                key="change_favorite_team_button",
                use_container_width=True,
            ):
                st.session_state["show_favorite_picker"] = True

            if st.session_state["show_favorite_picker"]:
                selected_favorite = st.selectbox(
                    "Choose your favorite team",
                    teams,
                    index=teams.index(favorite),
                    format_func=team_label,
                    key="favorite_team_picker",
                )

                save_col, cancel_col = st.columns(2)
                with save_col:
                    if st.button(
                        "Save",
                        type="primary",
                        key="save_favorite_team",
                        use_container_width=True,
                    ):
                        set_favorite_team(
                            current_user.email,
                            app_username,
                            selected_favorite,
                        )
                        st.session_state["show_favorite_picker"] = False
                        st.rerun()

                with cancel_col:
                    if st.button(
                        "Cancel",
                        key="cancel_favorite_team",
                        use_container_width=True,
                    ):
                        st.session_state["show_favorite_picker"] = False
                        st.rerun()

    weekly_upcoming = upcoming_games_for_week(schedules, current_season, active_week)
    featured_game, featured_result = game_of_week(bundle, weekly_upcoming)
    render_game_of_week(featured_game, featured_result)

    home_left, home_right = st.columns(2)
    with home_left:
        if teams:
            st.markdown(f"### Your team: {team_name(favorite)}")
            st.image(team_logo_url(favorite), width=90)
            record = team_record(bundle.team_games, favorite, current_season)
            st.metric("Record", f"{record['wins']}-{record['losses']}")
            render_injury_summary(injuries, favorite, compact=True)
            st.markdown("**Next game**")
            render_team_upcoming(favorite, bundle.schedules, 1)

    with home_right:
        st.markdown("### Weekly Challenge")
        my_score = user_score(current_user.email, schedules, challenge_stats)
        s1, s2, s3 = st.columns(3)
        s1.metric("Season points", my_score["total"])
        s2.metric("Winner-pick points", my_score["winner_points"])
        s3.metric("Stat points", my_score["stat_points"])

        if challenge_season_finished:
            st.success("Regular-season challenge complete. All weeks have been scored.")
        else:
            st.info(
                f"Week {active_challenge_week} is unlocked. Future weeks stay locked "
                "until every game in this week is final."
            )

        if challenge_completed_weeks:
            last_scored_week = max(challenge_completed_weeks)
            last_week_score = weekly_user_score(
                current_user.email,
                schedules,
                challenge_stats,
                current_season,
                last_scored_week,
            )
            st.caption(
                f"Week {last_scored_week} final score: {last_week_score['total']} points. "
                "Weekly points are added only after the entire week is complete."
            )
        else:
            st.caption(
                "Your season score stays at zero until every game in Week 1 is final."
            )

        standings = leaderboard(schedules, challenge_stats)
        if not standings.empty:
            st.markdown("**Leaderboard**")
            st.dataframe(standings.head(5), hide_index=True, use_container_width=True)
        else:
            st.info("No challenge entries yet. Make the first pick in Weekly Challenge.")


with tab_challenge:
    st.subheader(
        "Weekly Prediction Challenge"
        if challenge_season_finished
        else f"Weekly Prediction Challenge — Week {active_challenge_week}"
    )
    st.caption(
        "Only the current week is available. Future weeks unlock automatically after every "
        "game in the current week is final. The entire week's score is then added at once."
    )

    challenge_week = active_challenge_week
    week_games = games_for_week(schedules, current_season, challenge_week)

    score = user_score(current_user.email, schedules, challenge_stats)
    c1, c2, c3 = st.columns(3)
    c1.metric("Season points", score["total"])
    c2.metric("Scored winner picks", score["settled_winners"])
    c3.metric("Scored stat challenges", score["settled_stats"])

    if challenge_completed_weeks:
        previous_week = max(challenge_completed_weeks)
        previous_score = weekly_user_score(
            current_user.email,
            schedules,
            challenge_stats,
            current_season,
            previous_week,
        )
        with st.container(border=True):
            st.markdown(f"**Week {previous_week} final**")
            w1, w2, w3 = st.columns(3)
            w1.metric("Week points", previous_score["total"])
            w2.metric("Winner points", previous_score["winner_points"])
            w3.metric("Stat points", previous_score["stat_points"])

    if challenge_season_finished:
        st.success("The regular-season challenge is complete. No future weeks remain to unlock.")
    else:
        st.info(
            f"Week {challenge_week} is the only unlocked week. "
            f"Week {challenge_week + 1} stays locked until this entire week is final."
        )

    st.markdown("### Winner picks")
    if week_games.empty:
        st.info(f"No games found for Week {challenge_week}.")
    else:
        for _, game in week_games.iterrows():
            game_id = clean_text(game.get("game_id"))
            away = clean_text(game.get("away_team"))
            home = clean_text(game.get("home_team"))
            if not game_id or not away or not home:
                continue

            existing = winner_pick_for_user(current_user.email, game_id)
            winner = completed_winner(game)

            with st.container(border=True):
                g1, g2, g3 = st.columns([1, 1.3, 1])
                with g1:
                    st.image(team_logo_url(away), width=62)
                    st.markdown(f"**{team_name(away)}**")
                with g2:
                    st.markdown(
                        f"<div style='text-align:center;font-weight:700'>{format_kickoff(game)}</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(format_game_location(game))
                with g3:
                    st.image(team_logo_url(home), width=62)
                    st.markdown(f"**{team_name(home)}**")

                matchup_result, matchup_margin = matchup_projection(bundle, away, home)
                if matchup_margin:
                    st.markdown(
                        f"**Model projected margin:** {matchup_margin['label']}"
                    )
                    if matchup_result:
                        st.caption(
                            f"Model win probability: {team_name(away)} "
                            f"{matchup_result['away_probability']:.0%} • "
                            f"{team_name(home)} {matchup_result['home_probability']:.0%}. "
                            "Analytics estimate only; not a market line."
                        )

                if winner is not None:
                    pick_text = team_name(existing) if existing else "No pick"
                    result_text = "✅ Correct" if existing == winner else ("❌ Incorrect" if existing else "—")
                    st.caption(
                        f"Final winner: {team_name(winner)} • Your pick: {pick_text} • {result_text}"
                    )
                elif game_is_open(game) and not challenge_season_finished:
                    options = [away, home]
                    default_index = options.index(existing) if existing in options else 0
                    with st.form(f"winner_pick_{game_id}"):
                        pick = st.radio(
                            "Your winner",
                            options,
                            index=default_index,
                            format_func=team_name,
                            horizontal=True,
                        )
                        submitted = st.form_submit_button("Save pick")
                    if submitted:
                        save_winner_pick(
                            current_user.email,
                            app_username,
                            current_season,
                            challenge_week,
                            game_id,
                            pick,
                        )
                        st.success(f"Saved: {team_name(pick)}")
                else:
                    st.caption("Picks are locked because this game has started.")

    st.markdown("### Player-stat challenge")
    open_games = (
        week_games[week_games.apply(game_is_open, axis=1)].copy()
        if not week_games.empty and not challenge_season_finished
        else pd.DataFrame()
    )

    if open_games.empty or challenge_roster.empty:
        st.info("Player-stat challenges will be available when an upcoming game and roster data are available.")
    else:
        game_indexes = list(open_games.index)
        selected_game_index = st.selectbox(
            "Game",
            game_indexes,
            format_func=lambda idx: format_game_choice(open_games.loc[idx]),
            key="stat_game",
        )
        selected_game = open_games.loc[selected_game_index]
        stat_teams = [
            clean_text(selected_game.get("away_team")),
            clean_text(selected_game.get("home_team")),
        ]
        stat_team = st.selectbox(
            "Team",
            stat_teams,
            format_func=team_name,
            key="stat_team",
        )

        player_pool = challenge_roster[challenge_roster["team"] == stat_team].copy()
        player_pool = player_pool.sort_values(["position", "full_name"])

        if player_pool.empty:
            st.info("No eligible QB/RB/WR/TE players were found for this team.")
        else:
            player_keys = list(player_pool.index)
            selected_player_index = st.selectbox(
                "Player",
                player_keys,
                format_func=lambda idx: (
                    f"{player_pool.loc[idx, 'full_name']} ({player_pool.loc[idx, 'position']})"
                ),
                key="stat_player",
            )
            player = player_pool.loc[selected_player_index]
            position = clean_text(player.get("position")).upper()
            stat_options = POSITION_STATS.get(position, [])

            stat_key = st.selectbox(
                "Stat to predict",
                stat_options,
                format_func=lambda key: STAT_LABELS.get(key, key),
                key="stat_key",
            )
            predicted_value = st.number_input(
                "Your prediction",
                min_value=0.0,
                step=1.0,
                key="stat_value",
            )

            if st.button("Save stat prediction", type="primary"):
                save_stat_prediction(
                    current_user.email,
                    app_username,
                    current_season,
                    challenge_week,
                    clean_text(selected_game.get("game_id")),
                    clean_text(player.get("gsis_id")) or clean_text(player.get("full_name")),
                    clean_text(player.get("full_name")),
                    stat_key,
                    predicted_value,
                )
                st.success(
                    f"Saved: {player['full_name']} — {STAT_LABELS.get(stat_key, stat_key)} "
                    f"{predicted_value:g}"
                )

    saved_stats = stat_predictions_for_user(
        current_user.email,
        current_season,
        challenge_week,
    )
    if not saved_stats.empty:
        st.markdown("#### Your saved stat predictions")
        saved_display = saved_stats.copy()
        saved_display["Stat"] = saved_display["stat_key"].map(
            lambda key: STAT_LABELS.get(key, key)
        )
        saved_display = saved_display.rename(
            columns={
                "player_name": "Player",
                "predicted_value": "Prediction",
            }
        )
        st.dataframe(
            saved_display[["Player", "Stat", "Prediction"]],
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("### Leaderboard")
    standings = leaderboard(schedules, challenge_stats)
    if standings.empty:
        st.info("No leaderboard entries yet.")
    else:
        st.dataframe(standings, hide_index=True, use_container_width=True)


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
                predictor_margin = projected_margin_from_result(result, away, home)
                st.markdown(
                    f"**Model projected margin:** {predictor_margin['label']}"
                )
                st.caption("Approximate model estimate; not a market line.")
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


with tab_depth:
    st.subheader("Depth Chart")
    st.caption(
        "Latest available team depth chart arranged on the field. The larger name in each "
        "position card is the projected starter; the smaller name underneath is the next "
        "available depth option from the source."
    )

    depth_team = st.selectbox(
        "Team",
        teams,
        format_func=team_label,
        key="depth_chart_team",
    )

    depth_header_logo, depth_header_text = st.columns([0.12, 0.88], vertical_alignment="center")
    with depth_header_logo:
        st.image(team_logo_url(depth_team), width=92)
    with depth_header_text:
        st.markdown(f"## {team_name(depth_team)}")
        updated_label = latest_update_label(depth_charts, depth_team)
        if updated_label:
            st.caption(f"Latest depth-chart snapshot: {updated_label}")
        else:
            st.caption("Latest available depth-chart snapshot")

    depth_html = depth_chart_html(depth_charts, depth_team)
    if not depth_html:
        st.info(
            "A current depth chart is not available from the source for this team yet. "
            "It will populate automatically when the latest depth-chart data is published."
        )
    else:
        st.html(depth_html)
        st.caption(
            "Field placement is a Gridiron Central visualization of the published depth chart. "
            "Actual offensive and defensive formations change by package and play."
        )


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

                matchup_result, matchup_margin = matchup_projection(
                    bundle,
                    away_team,
                    home_team,
                )
                if matchup_margin:
                    st.markdown(
                        f"**Model projected margin:** {matchup_margin['label']}"
                    )
                    if matchup_result:
                        st.caption(
                            f"{team_name(away_team)} {matchup_result['away_probability']:.0%} • "
                            f"{team_name(home_team)} {matchup_result['home_probability']:.0%} "
                            "model win probability. Not a market line."
                        )

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
                "nflverse rosters. Trade data also includes a verified 2026 supplement for confirmed "
                "player trades that have not yet appeared in the upstream dataset."
            )


with tab_injuries:
    st.subheader("Player Status")
    st.caption(
        "Current league-wide injury and roster-availability tracker. The app refreshes the "
        "live status source every five minutes while in use and falls back to nflverse if "
        "the live source is unavailable. Official game designations can still change before kickoff."
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


with tab_map:
    st.subheader("NFL Franchise Map")
    st.caption(
        "Team logos mark each franchise's current home. Click a logo to open its history."
    )

    map_data = league_map_frame(TEAM_NAMES).copy()

    # Slightly separate teams that share the same stadium so both logos can be selected.
    map_offsets = {
        "LA": (-0.06, -0.06),
        "LAC": (0.06, 0.06),
        "NYG": (-0.06, -0.05),
        "NYJ": (0.06, 0.05),
    }

    map_records = []
    for _, row in map_data.iterrows():
        team_code = clean_text(row.get("team"))
        lat_offset, lon_offset = map_offsets.get(team_code, (0.0, 0.0))
        map_records.append(
            {
                "team": team_code,
                "name": team_name(team_code),
                "lat": float(row["lat"]) + lat_offset,
                "lon": float(row["lon"]) + lon_offset,
                "icon_data": {
                    "url": team_logo_url(team_code),
                    "width": 128,
                    "height": 128,
                    "anchorY": 128,
                },
            }
        )

    logo_layer = pdk.Layer(
        "IconLayer",
        data=map_records,
        id="team-logos",
        get_icon="icon_data",
        get_position="[lon, lat]",
        get_size=4,
        size_scale=12,
        size_min_pixels=38,
        size_max_pixels=64,
        pickable=True,
        auto_highlight=True,
    )

    logo_map = pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=38.3,
            longitude=-96.5,
            zoom=3.15,
            pitch=0,
        ),
        layers=[logo_layer],
        tooltip={"text": "{name}\nClick for franchise history"},
    )

    map_event = st.pydeck_chart(
        logo_map,
        height=520,
        selection_mode="single-object",
        on_select="rerun",
        key="nfl_franchise_logo_map",
    )

    if "league_map_team" not in st.session_state:
        st.session_state["league_map_team"] = teams[0] if teams else None
    if "map_team_fallback" not in st.session_state and teams:
        st.session_state["map_team_fallback"] = st.session_state["league_map_team"]

    try:
        selected_objects = map_event.selection.objects.get("team-logos", [])
        if selected_objects:
            selected_code = clean_text(selected_objects[0].get("team"))
            if selected_code in teams:
                st.session_state["league_map_team"] = selected_code
                st.session_state["map_team_fallback"] = selected_code
    except Exception:
        pass

    fallback_team = st.selectbox(
        "Or choose a team",
        teams,
        format_func=team_label,
        key="map_team_fallback",
    )
    if fallback_team != st.session_state.get("league_map_team"):
        st.session_state["league_map_team"] = fallback_team

    map_team = st.session_state.get("league_map_team")
    if map_team:
        history = team_history(map_team)

        st.divider()
        detail_logo, detail_main = st.columns([1, 3])
        with detail_logo:
            st.image(team_logo_url(map_team), width=150)

        with detail_main:
            st.markdown(f"## {team_name(map_team)}")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Founded", str(history.get("founded", "—")))
            h2.metric("Conference", history.get("conference", "—"))
            h3.metric("Division", history.get("division", "—"))
            h4.metric(
                "Franchise age",
                f"{current_season - int(history.get('founded', current_season))} years"
                if history.get("founded")
                else "—",
            )
            st.caption(
                f"{history.get('home', TEAM_HOME_LOCATIONS.get(map_team, ''))} • "
                f"{history.get('stadium', 'Current stadium')}"
            )

        st.markdown("### Franchise history")
        st.write(history.get("history", "Franchise history is being added."))

        history_left, history_right = st.columns([1.15, 1])
        with history_left:
            st.markdown("### Franchise legends — curated")
            legends = history.get("legends", [])
            if legends:
                st.markdown(
                    "\n".join(
                        f"{idx}. **{player}**"
                        for idx, player in enumerate(legends, start=1)
                    )
                )
                st.caption(
                    "This list is a curated franchise-history selection, not an official NFL ranking."
                )
            else:
                st.info("Franchise legends are still being added.")

        with history_right:
            st.markdown("### Current team snapshot")
            record = team_record(bundle.team_games, map_team, current_season)
            summary = current_season_summary(
                bundle.team_games,
                map_team,
                current_season,
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Record", f"{record['wins']}-{record['losses']}")
            m2.metric("Avg points", f"{summary['points_for']:.1f}")
            m3.metric("Avg point diff", f"{summary['point_diff']:+.1f}")

            st.markdown("**Next game**")
            render_team_upcoming(map_team, bundle.schedules, 1)
            st.markdown("**Player status**")
            render_injury_summary(injuries, map_team, compact=True)


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
