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
    default_home = teams.index("KC") if "KC" in teams else min(1, len(teams)-1)

    with c1:
        away = st.selectbox("Away team", teams, index=default_away)
    with c2:
        home = st.selectbox("Home team", teams, index=default_home)

    if away == home:
        st.warning("Choose two different teams.")
    else:
        result = predict_matchup(bundle, away, home)

        p1, p2, p3 = st.columns(3)
        p1.metric(f"{away} win probability", f"{result['away_probability']:.1%}")
        p2.metric(f"{home} win probability", f"{result['home_probability']:.1%}")
        p3.metric("Model pick", result["predicted_winner"])

        st.progress(result["confidence"])
        st.caption(
            f"Model confidence in its higher-probability side: {result['confidence']:.1%}. "
            "A probability is not a guarantee."
        )

        st.markdown("#### Biggest statistical drivers")
        factor_df = pd.DataFrame(result["factors"][:8])
        factor_df = factor_df[["category", "factor", "leans", "raw_difference", "model_contribution"]]
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
                    away: result["away_snapshot"][f],
                    home: result["home_snapshot"][f],
                }
            )
        st.dataframe(pd.DataFrame(profile_rows), hide_index=True, use_container_width=True)


with tab_upcoming:
    st.subheader("Upcoming schedule")
    upcoming = upcoming_games(bundle.schedules, 20)

    if upcoming.empty:
        st.info("No upcoming non-preseason games were found in the current schedule file.")
    else:
        st.dataframe(upcoming, hide_index=True, use_container_width=True)
        st.caption(
            "Use any matchup from this list in the Matchup Predictor. "
            "Final scores are incorporated after a data refresh."
        )


with tab_offseason:
    st.subheader("2026 offseason roster and draft layer")
    st.caption(
        "These are model inputs, not manual opinions. They are rebuilt from nflverse roster, "
        "draft, player-stat, and trade data when the model refreshes."
    )

    if bundle.offseason.empty:
        st.warning("Offseason data is not currently available from the source.")
    else:
        latest_season = int(bundle.offseason["season"].max())
        latest = bundle.offseason[bundle.offseason["season"] == latest_season].copy()

        display_cols = ["team"] + OFFSEASON_FEATURES
        latest = latest[display_cols]
        latest = latest.rename(columns={
            "team": "Team",
            **{f: OFFSEASON_DISPLAY_NAMES[f] for f in OFFSEASON_FEATURES},
        })

        st.dataframe(latest.sort_values("Team"), hide_index=True, use_container_width=True)

        st.markdown(
            """
            **How to read these values**

            - **Roster continuity** measures how much weighted personnel returned from the previous season.
            - **Veteran additions/departures** measure weighted roster turnover by position importance.
            - **Draft class impact** gives earlier picks and higher-impact positions more initial weight.
            - **Primary QB continuity** checks whether the previous season's leading passer is still on the roster.
            - **Trade additions/departures** count player movement recorded in the trade dataset.

            These features do not automatically make a team better or worse. The logistic-regression
            model learns from historical seasons how much each difference has actually mattered.
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
