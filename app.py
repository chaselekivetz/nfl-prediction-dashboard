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

MODEL_CACHE_VERSION = "weather-schema-v1"


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
            "Impact score": round(float(player.get("impact_score", 0.0)), 2),
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
        f"**Top move:** {top['Player']} · {top['Position']}  "
        f"<span class='small-note'>Impact {top['Impact score']:.2f}</span>",
        unsafe_allow_html=True,
    )
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        height=min(420, 78 + (len(table) * 35)),
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

except Exception as exc:
    st.error("The dashboard could not load or train the NFL model.")
    st.exception(exc)
    st.stop()


teams = available_teams(bundle)

tab_predict, tab_upcoming, tab_offseason, tab_accuracy, tab_method = st.tabs(
    ["Matchup Predictor", "Upcoming Games", "Offseason Changes", "Model Accuracy", "How It Works"]
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

        p1, p2, p3 = st.columns(3)
        p1.metric(f"{away_name} win probability", f"{result['away_probability']:.1%}")
        p2.metric(f"{home_name} win probability", f"{result['home_probability']:.1%}")
        p3.metric("Model pick", predicted_name)

        st.progress(result["confidence"])
        st.caption(
            f"Model confidence in its higher-probability side: {result['confidence']:.1%}. "
            "A probability is not a guarantee."
        )

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
                    "Impact rank uses positional importance plus prior-season usage when available. "
                    "It is a relative roster-impact score, not a salary or contract-value ranking."
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
