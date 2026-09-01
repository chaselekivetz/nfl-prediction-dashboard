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
    "team performance; it is not a wagering recommendation."
)

@st.cache_data(ttl=900, show_spinner=False)
def get_data():
    # 2023-2025 are the initial three historical seasons.
    # Completed 2026 games are automatically included when nflverse publishes them.
    return load_nfl_data([2023, 2024, 2025, 2026])


@st.cache_resource(show_spinner=False)
def get_model(_schedules, _team_stats, data_signature):
    # data_signature changes when the number of completed schedule rows changes,
    # causing the model to retrain after new final results arrive.
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
        When a new final score appears in nflverse, the next refresh rebuilds
        the dataset and retrains the logistic-regression model.
        """
    )

try:
    with st.spinner("Loading NFL data and training model..."):
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

tab_predict, tab_upcoming, tab_accuracy, tab_method = st.tabs(
    ["Matchup Predictor", "Upcoming Games", "Model Accuracy", "How It Works"]
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
        factor_df = pd.DataFrame(result["factors"][:6])
        factor_df = factor_df[["factor", "leans", "raw_difference", "model_contribution"]]
        factor_df.columns = [
            "Factor",
            "Leans toward",
            "Home − away difference",
            "Model contribution",
        ]
        st.dataframe(factor_df, hide_index=True, use_container_width=True)

        st.markdown("#### Current rolling team profile")
        profile_rows = []
        for f in FEATURES:
            profile_rows.append(
                {
                    "Metric": DISPLAY_NAMES[f],
                    away: result["away_snapshot"][f],
                    home: result["home_snapshot"][f],
                }
            )
        st.dataframe(
            pd.DataFrame(profile_rows),
            hide_index=True,
            use_container_width=True,
        )


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
        This version uses **logistic regression**, which is a good first machine-learning
        model because its behavior is understandable and its inputs can be inspected.

        For every historical matchup, the program creates pregame rolling statistics.
        It uses only earlier games, then compares the home and away teams.

        **Tracked signals**
        - 8-game point differential
        - 8-game points scored
        - 8-game points allowed
        - 8-game offensive yards
        - 8-game yards allowed
        - 8-game turnover margin
        - last-5 win percentage
        - home/away performance
        - opponent strength
        - recency weighting by season

        The final model is retrained on all completed games from 2023 onward. Older
        seasons receive slightly less weight than newer seasons. As 2026 final scores
        arrive, those games become new training examples.
        """
    )

    st.info(
        "Important model limitation: roster changes, injuries, quarterback changes, "
        "weather, coaching changes, and other context are not yet included. Those are "
        "good candidates for a later version."
    )
