import streamlit as st
import pandas as pd

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


st.set_page_config(
    page_title="NFL Prediction Lab",
    page_icon="🏈",
    layout="wide",
)

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
def get_model(_schedules, _team_stats, data_signature):
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
        bundle = get_model(schedules, team_stats, signature)

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
    upcoming = upcoming_games(bundle.schedules, 20)

    if upcoming.empty:
        st.info("No upcoming non-preseason games were found in the current schedule file.")
    else:
        upcoming_display = upcoming.copy()
        for column in ["away_team", "home_team", "Away", "Home"]:
            if column in upcoming_display.columns:
                upcoming_display[column] = upcoming_display[column].map(
                    lambda value: team_name(value) if value in TEAM_NAMES else value
                )
        st.dataframe(upcoming_display, hide_index=True, use_container_width=True)
        st.caption(
            "Use any matchup from this list in the Matchup Predictor. "
            "Final scores are incorporated after a data refresh."
        )


with tab_offseason:
    st.subheader("2026 offseason roster, trade, and draft changes")
    st.caption(
        "The model still uses numeric roster, draft, and trade features behind the scenes. "
        "This view shows the actual player movement that produced those inputs."
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
            selected_logo = st.empty()
            selected_team = st.selectbox(
                f"{latest_season} team",
                offseason_teams,
                format_func=team_label,
                key="offseason_team",
            )
            show_team_logo(selected_logo, selected_team)

            team_row = latest[latest["team"] == selected_team].iloc[-1]
            st.markdown(f"### {team_name(selected_team)}")

            a1, a2 = st.columns(2)
            with a1:
                show_player_list(
                    "Players acquired / added",
                    team_row.get("added_players", ""),
                    "No roster additions were found in the current source data.",
                )
            with a2:
                show_player_list(
                    "Players moved away / departed",
                    team_row.get("departed_players", ""),
                    "No roster departures were found in the current source data.",
                )

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

            st.markdown("#### Model offseason inputs")
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
                "Roster additions and departures are found by comparing the prior-season and "
                "current-season nflverse rosters. Draft and trade names come from nflverse datasets."
            )

        st.markdown(
            """
            **How the model uses offseason changes**

            - **Roster continuity** measures how much weighted personnel returned from the previous season.
            - **Veteran additions/departures** measure weighted roster turnover by position importance.
            - **Draft class impact** gives earlier picks and higher-impact positions more initial weight.
            - **Primary QB continuity** checks whether the previous season's leading passer is still on the roster.
            - **Trade additions/departures** measure player movement recorded in the trade dataset.

            The player names are there to make the data transparent. The logistic-regression model
            continues to train on the numeric features rather than treating every player as equally valuable.
            """
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
