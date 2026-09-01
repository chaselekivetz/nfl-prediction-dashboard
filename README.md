# NFL Prediction Lab

A browser-based educational NFL analytics project built with Streamlit.

## What it does

- Loads NFL schedule/results and weekly team statistics from nflverse through `nflreadpy`.
- Starts with the 2023, 2024, and 2025 seasons.
- Includes completed 2026 games whenever they become available in the source data.
- Builds pregame rolling features without using the current game's result.
- Trains an easy-to-understand logistic regression model.
- Produces home/away win probabilities.
- Shows the factors that influenced the prediction most.
- Uses a chronological validation split to report model accuracy and Brier score.
- Refreshes cached NFL data every 15 minutes while the dashboard is being used.
- Retrains when completed-game data changes.

## Tracked statistics

- Recent point differential
- Points scored
- Points allowed
- Offensive yards
- Yards allowed
- Turnover margin
- Last-5 win percentage
- Home/away performance
- Opponent strength
- Season recency weighting

## Run it

1. Install Python 3.11+.
2. Open a terminal in this folder.
3. Create a virtual environment if desired.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Launch:

```bash
streamlit run app.py
```

A browser window should open automatically.

## Updating scores

The app uses current nflverse schedule/results data. The local cache expires every
15 minutes while the app is being used. The **Refresh NFL data** button forces a fresh
download immediately. When newly completed games appear, the model is rebuilt and those
final scores become additional training examples.

This is not a background service: if the dashboard is closed, it does not continuously
run on its own.

## Model note

This is intentionally a beginner-friendly model. A future version could add quarterback
status, injuries, weather, EPA/play, success rate, rest days, coaching changes, and a
walk-forward backtest.

## Responsible-use note

This project is for sports analytics and machine-learning practice. Win probabilities
are uncertain estimates and are not wagering recommendations.
