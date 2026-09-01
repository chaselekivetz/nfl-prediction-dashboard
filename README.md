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


## Private access and invitations

The dashboard now **fails closed**: it stops before loading NFL data unless OIDC authentication is configured and the signed-in email is authorized.

The access layer uses:

- **OpenID Connect** through Streamlit `st.login()` to identify the user. Auth0 email sign-in is recommended for the administrator account.
- **Supabase** for the persistent invite/allowlist table.
- **Resend** for invitation emails.
- **Streamlit Secrets** for every credential. Do not commit real secrets to GitHub.

### 1. Create the allowlist table

Create a Supabase project, open its SQL editor, and run `setup_access.sql` from this repository.

The table has no public RLS policies. The app uses a Supabase server-side Secret key in Streamlit Secrets.

### 2. Configure Auth0 email sign-in

Create an Auth0 application and configure email passwordless sign-in (one-time code or magic link). Add this callback URL:

```
https://YOUR-APP.streamlit.app/oauth2callback
```

Your Auth0 OIDC metadata URL will look like:

```
https://YOUR-AUTH0-DOMAIN/.well-known/openid-configuration
```

### 3. Configure Streamlit Secrets

Open the deployed app's **Settings → Secrets** and copy the structure from:

```
.streamlit/secrets.example.toml
```

Replace every placeholder with the real value. The email in `admin_emails` is the administrator account that can always enter the app and manage invitations.

Generate a long random `cookie_secret`. Keep the Auth0 client secret, Supabase Secret key, and Resend API key only in Streamlit Secrets.

### 4. Configure invitation email

Create a Resend sending API key and verified sender/domain, then fill in:

- `resend_api_key`
- `from_email`
- `app_url`

If Resend is not configured yet, the admin can still approve an email; the app will show the URL so it can be shared manually.

### 5. Invite users

After signing in as an administrator, open **Admin access** in the sidebar.

Enter an email and click **Approve & send invite**. The user is written to the persistent allowlist and receives the dashboard link. They must sign in with the same approved email.

The administrator can also revoke an invited user from the same panel.

### Security behavior

- An unauthenticated visitor sees only the sign-in screen.
- An authenticated but unapproved account sees only an access-denied screen.
- NFL schedules, model training, predictions, and offseason data are not loaded until authorization succeeds.
- Database or authorization-check failures deny access rather than allowing it.
- Real secrets are excluded from Git with `.gitignore`.
