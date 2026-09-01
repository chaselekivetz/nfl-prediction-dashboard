from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import nflreadpy as nfl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from offseason import (
    OFFSEASON_DISPLAY_NAMES,
    OFFSEASON_FEATURES,
    build_offseason_features,
    team_offseason_snapshot,
)
from weather import (
    WEATHER_DISPLAY_NAMES,
    WEATHER_GAME_COLUMNS,
    WEATHER_MODEL_FEATURES,
    WEATHER_TEAM_FEATURES,
    add_schedule_weather,
    add_team_weather_history,
    model_weather_values,
)


BASE_FEATURES = [
    "point_diff_form",
    "offense_points_form",
    "defense_points_form",
    "yards_for_form",
    "yards_allowed_form",
    "turnover_margin_form",
    "recent_win_pct",
    "location_win_pct",
    "opponent_strength",
]

FEATURES = BASE_FEATURES + OFFSEASON_FEATURES

DISPLAY_NAMES = {
    "point_diff_form": "Recent point differential",
    "offense_points_form": "Recent scoring",
    "defense_points_form": "Recent points allowed",
    "yards_for_form": "Recent offensive yards",
    "yards_allowed_form": "Recent yards allowed",
    "turnover_margin_form": "Recent turnover margin",
    "recent_win_pct": "Last-5 win rate",
    "location_win_pct": "Home/away performance",
    "opponent_strength": "Opponent strength",
    **OFFSEASON_DISPLAY_NAMES,
}

SEASON_WEIGHTS = {
    2023: 0.60,
    2024: 0.75,
    2025: 0.90,
    2026: 1.00,
}

OFFSEASON_DEFAULTS = {
    "roster_continuity": 0.65,
    "addition_weight": 0.25,
    "departure_weight": 0.25,
    "draft_impact": 1.0,
    "qb_returning": 0.5,
    "trade_additions": 0.0,
    "trade_departures": 0.0,
}


@dataclass
class ModelBundle:
    model: Pipeline
    training_games: pd.DataFrame
    team_games: pd.DataFrame
    schedules: pd.DataFrame
    offseason: pd.DataFrame
    validation_accuracy: float | None
    validation_brier: float | None


def _to_pandas(frame):
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return pd.DataFrame(frame)


def load_nfl_data(seasons: Iterable[int] = (2023, 2024, 2025, 2026)):
    seasons = list(seasons)
    schedules = _to_pandas(nfl.load_schedules(seasons))

    stat_frames = []
    unavailable_seasons = []
    for season in seasons:
        try:
            frame = _to_pandas(nfl.load_team_stats([season], summary_level="week"))
            if not frame.empty:
                stat_frames.append(frame)
        except Exception:
            unavailable_seasons.append(season)

    if not stat_frames:
        raise ValueError("No NFL weekly team-stat files were available for the requested seasons.")

    team_stats = pd.concat(stat_frames, ignore_index=True, sort=False)
    team_stats.attrs["unavailable_seasons"] = unavailable_seasons
    return schedules, team_stats


def _first_existing(df: pd.DataFrame, names, default=0.0):
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(default, index=df.index, dtype=float)


def _sum_existing(df: pd.DataFrame, names):
    result = pd.Series(0.0, index=df.index)
    found = False
    for name in names:
        if name in df.columns:
            result = result + pd.to_numeric(df[name], errors="coerce").fillna(0.0)
            found = True
    if not found:
        return pd.Series(0.0, index=df.index)
    return result


def normalize_team_stats(team_stats: pd.DataFrame) -> pd.DataFrame:
    stats = team_stats.copy()
    if "season_type" in stats.columns:
        stats = stats[stats["season_type"].isin(["REG", "POST"])].copy()

    required = {"game_id", "team"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"Team-stat data is missing required columns: {sorted(missing)}")

    stats["yards_for"] = _first_existing(stats, ["passing_yards"]) + _first_existing(stats, ["rushing_yards"])
    stats["turnovers_lost"] = _first_existing(stats, ["passing_interceptions"]) + _sum_existing(
        stats,
        [
            "rushing_fumbles_lost",
            "receiving_fumbles_lost",
            "sack_fumbles_lost",
            "special_teams_fumbles_lost",
        ],
    )

    return stats[["game_id", "team", "yards_for", "turnovers_lost"]].drop_duplicates(["game_id", "team"])


def completed_games(schedules: pd.DataFrame) -> pd.DataFrame:
    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"] != "PRE"].copy()

    required = {"game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(s.columns)
    if missing:
        raise ValueError(f"Schedule data is missing required columns: {sorted(missing)}")

    s["home_score"] = pd.to_numeric(s["home_score"], errors="coerce")
    s["away_score"] = pd.to_numeric(s["away_score"], errors="coerce")
    s = s[s["home_score"].notna() & s["away_score"].notna()].copy()

    if "gameday" in s.columns:
        s["gameday"] = pd.to_datetime(s["gameday"], errors="coerce")
    else:
        s["gameday"] = pd.to_datetime(s["season"].astype(str) + "-01-01", errors="coerce") + pd.to_timedelta(
            pd.to_numeric(s["week"], errors="coerce").fillna(0) * 7, unit="D"
        )

    s = add_schedule_weather(s)
    return s.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def build_team_games(schedules: pd.DataFrame, team_stats: pd.DataFrame, offseason: pd.DataFrame) -> pd.DataFrame:
    games = completed_games(schedules)
    stats = normalize_team_stats(team_stats)

    base_game_cols = [
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
    ] + WEATHER_GAME_COLUMNS

    home = games[base_game_cols].copy()
    home = home.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_score": "points_for",
        "away_score": "points_against",
    })
    home["is_home"] = 1

    away = games[base_game_cols].copy()
    away = away.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_score": "points_for",
        "home_score": "points_against",
    })
    away["is_home"] = 0

    tg = pd.concat([home, away], ignore_index=True)
    tg["win"] = (tg["points_for"] > tg["points_against"]).astype(int)
    tg["point_diff"] = tg["points_for"] - tg["points_against"]
    tg = tg.merge(stats, on=["game_id", "team"], how="left")

    opp_stats = stats.rename(columns={
        "team": "opponent",
        "yards_for": "yards_allowed",
        "turnovers_lost": "opponent_turnovers_lost",
    })
    tg = tg.merge(
        opp_stats[["game_id", "opponent", "yards_allowed", "opponent_turnovers_lost"]],
        on=["game_id", "opponent"],
        how="left",
    )

    tg["yards_for"] = tg["yards_for"].fillna(0.0)
    tg["yards_allowed"] = tg["yards_allowed"].fillna(0.0)
    tg["turnovers_lost"] = tg["turnovers_lost"].fillna(0.0)
    tg["opponent_turnovers_lost"] = tg["opponent_turnovers_lost"].fillna(0.0)
    tg["turnover_margin"] = tg["opponent_turnovers_lost"] - tg["turnovers_lost"]

    if not offseason.empty:
        tg = tg.merge(offseason[["season", "team"] + OFFSEASON_FEATURES], on=["season", "team"], how="left")
    for feature in OFFSEASON_FEATURES:
        if feature not in tg.columns:
            tg[feature] = OFFSEASON_DEFAULTS[feature]
        tg[feature] = pd.to_numeric(tg[feature], errors="coerce").fillna(OFFSEASON_DEFAULTS[feature])

    tg = tg.sort_values(["gameday", "game_id", "team"]).reset_index(drop=True)

    def rolling_prior(series, window):
        return series.shift(1).rolling(window, min_periods=1).mean()

    grouped = tg.groupby("team", group_keys=False)
    tg["point_diff_form"] = grouped["point_diff"].transform(lambda x: rolling_prior(x, 8))
    tg["offense_points_form"] = grouped["points_for"].transform(lambda x: rolling_prior(x, 8))
    tg["defense_points_form"] = grouped["points_against"].transform(lambda x: rolling_prior(x, 8))
    tg["yards_for_form"] = grouped["yards_for"].transform(lambda x: rolling_prior(x, 8))
    tg["yards_allowed_form"] = grouped["yards_allowed"].transform(lambda x: rolling_prior(x, 8))
    tg["turnover_margin_form"] = grouped["turnover_margin"].transform(lambda x: rolling_prior(x, 8))
    tg["recent_win_pct"] = grouped["win"].transform(lambda x: rolling_prior(x, 5))
    tg["prior_win_pct"] = grouped["win"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    tg["location_win_pct"] = tg.groupby(["team", "is_home"])["win"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean()
    )

    tg = add_team_weather_history(tg)

    opp_strength = tg[["game_id", "team", "prior_win_pct"]].rename(
        columns={"team": "opponent", "prior_win_pct": "opponent_strength"}
    )
    return tg.merge(opp_strength, on=["game_id", "opponent"], how="left")


def build_training_games(team_games: pd.DataFrame) -> pd.DataFrame:
    home = team_games[team_games["is_home"] == 1].copy()
    away = team_games[team_games["is_home"] == 0].copy()

    weather_cols = WEATHER_TEAM_FEATURES + WEATHER_GAME_COLUMNS
    home_cols = ["game_id", "season", "week", "gameday", "team", "opponent", "win"] + FEATURES + weather_cols
    away_cols = ["game_id", "team"] + FEATURES + weather_cols

    home = home[home_cols].rename(
        columns={"team": "home_team", "opponent": "away_team", "win": "home_win"}
    )
    away = away[away_cols].rename(columns={"team": "away_team"})
    merged = home.merge(away, on=["game_id", "away_team"], suffixes=("_home", "_away"))

    for feature in FEATURES:
        merged[f"diff_{feature}"] = merged[f"{feature}_home"] - merged[f"{feature}_away"]

    # Weather severity is shared by both teams in a game. Team edges compare
    # each club's prior point-differential performance in similar conditions.
    merged["weather_cold_severity"] = merged["cold_severity_home"]
    merged["weather_wind_severity"] = merged["wind_severity_home"]
    merged["weather_heat_severity"] = merged["heat_severity_home"]
    merged["weather_outdoor"] = merged["outdoor_game_home"] * merged["weather_known_home"]
    merged["weather_cold_edge"] = merged["cold_severity_home"] * (
        merged["weather_cold_form_home"] - merged["weather_cold_form_away"]
    )
    merged["weather_wind_edge"] = merged["wind_severity_home"] * (
        merged["weather_wind_form_home"] - merged["weather_wind_form_away"]
    )
    merged["weather_heat_edge"] = merged["heat_severity_home"] * (
        merged["weather_heat_form_home"] - merged["weather_heat_form_away"]
    )

    diff_features = [f"diff_{f}" for f in FEATURES]
    model_features = diff_features + WEATHER_MODEL_FEATURES
    merged = merged.dropna(subset=model_features + ["home_win"]).copy()
    return merged.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def train_model(schedules: pd.DataFrame, team_stats: pd.DataFrame) -> ModelBundle:
    seasons = sorted(
        pd.to_numeric(schedules["season"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    offseason = build_offseason_features(seasons)
    tg = build_team_games(schedules, team_stats, offseason)
    train_games = build_training_games(tg)

    diff_features = [f"diff_{f}" for f in FEATURES]
    model_features = diff_features + WEATHER_MODEL_FEATURES

    if len(train_games) < 50:
        raise ValueError("Not enough completed games with rolling features to train the model.")

    split = max(1, int(len(train_games) * 0.80))
    split = min(split, len(train_games) - 1)
    older = train_games.iloc[:split].copy()
    newer = train_games.iloc[split:].copy()

    validation_model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    older_weights = older["season"].map(SEASON_WEIGHTS).fillna(1.0).to_numpy()
    validation_model.fit(
        older[model_features],
        older["home_win"],
        logit__sample_weight=older_weights,
    )

    val_prob = validation_model.predict_proba(newer[model_features])[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    val_accuracy = accuracy_score(newer["home_win"], val_pred)
    val_brier = brier_score_loss(newer["home_win"], val_prob)

    final_model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    all_weights = train_games["season"].map(SEASON_WEIGHTS).fillna(1.0).to_numpy()
    final_model.fit(
        train_games[model_features],
        train_games["home_win"],
        logit__sample_weight=all_weights,
    )

    return ModelBundle(
        model=final_model,
        training_games=train_games,
        team_games=tg,
        schedules=schedules,
        offseason=offseason,
        validation_accuracy=float(val_accuracy),
        validation_brier=float(val_brier),
    )


def _latest_snapshot(bundle: ModelBundle, team: str, is_home: int) -> dict:
    history = bundle.team_games[bundle.team_games["team"] == team].sort_values(["gameday", "game_id"])
    if history.empty:
        raise ValueError(f"No completed-game history found for {team}.")

    last8 = history.tail(8)
    last5 = history.tail(5)
    loc = history[history["is_home"] == is_home].tail(12)
    current_win_pct = float(history["win"].mean())

    snapshot = {
        "point_diff_form": float(last8["point_diff"].mean()),
        "offense_points_form": float(last8["points_for"].mean()),
        "defense_points_form": float(last8["points_against"].mean()),
        "yards_for_form": float(last8["yards_for"].mean()),
        "yards_allowed_form": float(last8["yards_allowed"].mean()),
        "turnover_margin_form": float(last8["turnover_margin"].mean()),
        "recent_win_pct": float(last5["win"].mean()),
        "location_win_pct": float(loc["win"].mean()) if not loc.empty else current_win_pct,
        "opponent_strength": current_win_pct,
    }

    current_season = int(pd.to_numeric(bundle.schedules["season"], errors="coerce").max())
    offseason = team_offseason_snapshot(bundle.offseason, team, current_season)
    for feature in OFFSEASON_FEATURES:
        snapshot[feature] = offseason.get(feature, OFFSEASON_DEFAULTS[feature])

    latest = history.iloc[-1]
    for feature in WEATHER_TEAM_FEATURES:
        snapshot[feature] = float(latest.get(feature, 0.0))

    return snapshot


def predict_matchup(
    bundle: ModelBundle,
    away_team: str,
    home_team: str,
    weather: dict | None = None,
):
    if away_team == home_team:
        raise ValueError("Choose two different teams.")

    home = _latest_snapshot(bundle, home_team, is_home=1)
    away = _latest_snapshot(bundle, away_team, is_home=0)

    home["opponent_strength"] = float(
        bundle.team_games[bundle.team_games["team"] == away_team]["win"].mean()
    )
    away["opponent_strength"] = float(
        bundle.team_games[bundle.team_games["team"] == home_team]["win"].mean()
    )

    row = {f"diff_{f}": home[f] - away[f] for f in FEATURES}
    row.update(model_weather_values(weather, home, away))

    diff_features = [f"diff_{f}" for f in FEATURES]
    model_features = diff_features + WEATHER_MODEL_FEATURES
    X = pd.DataFrame([row], columns=model_features)

    home_prob = float(bundle.model.predict_proba(X)[0, 1])
    away_prob = 1.0 - home_prob

    baseline_row = row.copy()
    for feature in WEATHER_MODEL_FEATURES:
        baseline_row[feature] = 0.0
    baseline_X = pd.DataFrame([baseline_row], columns=model_features)
    baseline_home_prob = float(bundle.model.predict_proba(baseline_X)[0, 1])

    scaler = bundle.model.named_steps["scale"]
    logit = bundle.model.named_steps["logit"]
    scaled = scaler.transform(X)[0]
    coefs = logit.coef_[0]

    contributions = []
    for feature, value, contribution in zip(model_features, X.iloc[0], scaled * coefs):
        if feature.startswith("diff_"):
            original = feature[5:]
            factor_name = DISPLAY_NAMES[original]
            category = "Offseason" if original in OFFSEASON_FEATURES else "Performance"
        else:
            factor_name = WEATHER_DISPLAY_NAMES[feature]
            category = "Weather"

        contributions.append({
            "factor": factor_name,
            "raw_difference": float(value),
            "model_contribution": float(contribution),
            "leans": home_team if contribution >= 0 else away_team,
            "strength": abs(float(contribution)),
            "category": category,
        })

    contributions.sort(key=lambda x: x["strength"], reverse=True)

    return {
        "away_team": away_team,
        "home_team": home_team,
        "away_probability": away_prob,
        "home_probability": home_prob,
        "baseline_home_probability": baseline_home_prob,
        "baseline_away_probability": 1.0 - baseline_home_prob,
        "weather_delta_home": home_prob - baseline_home_prob,
        "predicted_winner": home_team if home_prob >= 0.5 else away_team,
        "confidence": max(home_prob, away_prob),
        "factors": contributions,
        "home_snapshot": home,
        "away_snapshot": away,
        "weather": weather or {},
    }


def available_teams(bundle: ModelBundle):
    return sorted(bundle.team_games["team"].dropna().astype(str).unique().tolist())


def upcoming_games(schedules: pd.DataFrame, limit: int = 16):
    s = schedules.copy()
    if "game_type" in s.columns:
        s = s[s["game_type"] != "PRE"].copy()

    s["home_score"] = pd.to_numeric(s.get("home_score"), errors="coerce")
    s["away_score"] = pd.to_numeric(s.get("away_score"), errors="coerce")
    future = s[s["home_score"].isna() | s["away_score"].isna()].copy()

    if "gameday" in future.columns:
        future["gameday"] = pd.to_datetime(future["gameday"], errors="coerce")
        future = future.sort_values(["gameday", "game_id"])

    keep = [
        c for c in
        ["season", "week", "gameday", "gametime", "away_team", "home_team", "roof", "stadium"]
        if c in future.columns
    ]
    return future[keep].head(limit).reset_index(drop=True)
