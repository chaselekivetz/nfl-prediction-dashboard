from __future__ import annotations

from dataclasses import dataclass
import html
import re

import requests
import streamlit as st


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@dataclass(frozen=True)
class AccessUser:
    email: str
    name: str
    is_admin: bool


def _section(name: str):
    if name not in st.secrets:
        return {}
    return st.secrets[name]


def _normalise_email(value: str) -> str:
    return str(value or "").strip().lower()


def _email_list(value) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        value = [value]
    return {_normalise_email(item) for item in value if _normalise_email(item)}


def _auth_is_configured() -> bool:
    auth = _section("auth")
    required = [
        "redirect_uri",
        "cookie_secret",
        "client_id",
        "client_secret",
        "server_metadata_url",
    ]
    return all(str(auth.get(key, "")).strip() for key in required)


def _access_config():
    return _section("access")


def _admin_emails() -> set[str]:
    return _email_list(_access_config().get("admin_emails", []))


def _static_approved_emails() -> set[str]:
    return _email_list(_access_config().get("approved_emails", []))


def _database_is_configured() -> bool:
    access = _access_config()
    return bool(
        str(access.get("supabase_url", "")).strip()
        and str(access.get("supabase_service_role_key", "")).strip()
    )


def _supabase_headers(prefer: str | None = None) -> dict[str, str]:
    access = _access_config()
    key = str(access.get("supabase_service_role_key", "")).strip()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _supabase_endpoint() -> str:
    base = str(_access_config().get("supabase_url", "")).strip().rstrip("/")
    return f"{base}/rest/v1/app_users"


def _is_database_approved(email: str) -> bool:
    if not _database_is_configured():
        return False

    response = requests.get(
        _supabase_endpoint(),
        headers=_supabase_headers(),
        params={
            "select": "email,active",
            "email": f"eq.{email}",
            "active": "eq.true",
            "limit": "1",
        },
        timeout=10,
    )
    response.raise_for_status()
    return bool(response.json())


def _is_approved(email: str) -> bool:
    if email in _admin_emails():
        return True
    if email in _static_approved_emails():
        return True
    try:
        return _is_database_approved(email)
    except requests.RequestException:
        # Fail closed if the persistent allowlist cannot be checked.
        return False


def require_access() -> AccessUser:
    """Stop the app before loading NFL data unless the signed-in user is approved."""
    if not _auth_is_configured():
        st.title("🔒 NFL Prediction Lab")
        st.error("Private access is not configured yet.")
        st.info(
            "The dashboard is intentionally locked until OIDC authentication is configured "
            "in Streamlit Secrets. See the Access setup section in README.md."
        )
        st.stop()

    if not st.user.is_logged_in:
        st.title("🔒 NFL Prediction Lab")
        st.write("This dashboard is private. Sign in with an approved account to continue.")
        st.button("Sign in", type="primary", on_click=st.login)
        st.stop()

    claims = st.user.to_dict()
    email = _normalise_email(claims.get("email", ""))
    if not email:
        preferred = _normalise_email(claims.get("preferred_username", ""))
        if "@" in preferred:
            email = preferred

    if not email:
        st.error("Your identity provider did not return an email address.")
        st.button("Log out", on_click=st.logout)
        st.stop()

    if claims.get("email_verified") is False:
        st.error("Your email address must be verified before this dashboard can be opened.")
        st.button("Log out", on_click=st.logout)
        st.stop()

    if not _is_approved(email):
        st.title("🔒 NFL Prediction Lab")
        st.warning("This account has not been invited to the dashboard.")
        st.caption(email)
        st.button("Use a different account", on_click=st.logout)
        st.stop()

    name = str(claims.get("name") or email).strip()
    return AccessUser(email=email, name=name, is_admin=email in _admin_emails())


def _validate_email(value: str) -> str:
    email = _normalise_email(value)
    if not EMAIL_RE.match(email):
        raise ValueError("Enter a valid email address.")
    return email


def _approve_user(email: str, invited_by: str) -> None:
    if not _database_is_configured():
        raise RuntimeError("Persistent access storage is not configured.")

    payload = {
        "email": email,
        "active": True,
        "invited_by": invited_by,
    }
    response = requests.post(
        _supabase_endpoint(),
        headers=_supabase_headers("resolution=merge-duplicates,return=representation"),
        params={"on_conflict": "email"},
        json=payload,
        timeout=10,
    )
    response.raise_for_status()


def _revoke_user(email: str) -> None:
    if not _database_is_configured():
        raise RuntimeError("Persistent access storage is not configured.")

    response = requests.patch(
        _supabase_endpoint(),
        headers=_supabase_headers("return=minimal"),
        params={"email": f"eq.{email}"},
        json={"active": False},
        timeout=10,
    )
    response.raise_for_status()


def _list_users() -> list[dict]:
    if not _database_is_configured():
        return []

    response = requests.get(
        _supabase_endpoint(),
        headers=_supabase_headers(),
        params={
            "select": "email,active,invited_at,invited_by",
            "order": "invited_at.desc",
        },
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def _send_invite_email(email: str, invited_by: str) -> tuple[bool, str]:
    access = _access_config()
    api_key = str(access.get("resend_api_key", "")).strip()
    from_email = str(access.get("from_email", "")).strip()
    app_url = str(access.get("app_url", "")).strip()

    if not api_key or not from_email or not app_url:
        return False, "Access was approved, but invite-email delivery is not configured yet."

    safe_email = html.escape(email)
    safe_inviter = html.escape(invited_by)
    safe_url = html.escape(app_url, quote=True)

    response = requests.post(
        "https://api.resend.com/emails",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "from": from_email,
            "to": [email],
            "subject": "You're invited to NFL Prediction Lab",
            "html": (
                "<h2>You're invited to NFL Prediction Lab</h2>"
                f"<p>{safe_inviter} approved {safe_email} for private access.</p>"
                f'<p><a href="{safe_url}">Open NFL Prediction Lab</a></p>'
                "<p>Sign in with the same email address that received this invitation.</p>"
            ),
            "tags": [{"name": "category", "value": "nfl_dashboard_invite"}],
        },
        timeout=10,
    )
    response.raise_for_status()
    return True, "Invite email sent."


def invite_system_status() -> dict:
    access = _access_config()
    return {
        "persistent_storage": _database_is_configured(),
        "email_delivery": bool(
            str(access.get("resend_api_key", "")).strip()
            and str(access.get("from_email", "")).strip()
            and str(access.get("app_url", "")).strip()
        ),
        "app_url": str(access.get("app_url", "")).strip(),
    }


def invite_user(email: str, invited_by: str) -> dict:
    email = _validate_email(email)
    if email in _admin_emails():
        return {
            "email": email,
            "approved": True,
            "email_sent": False,
            "message": "That email is already an administrator.",
        }

    _approve_user(email, invited_by)
    sent, message = _send_invite_email(email, invited_by)
    return {
        "email": email,
        "approved": True,
        "email_sent": bool(sent),
        "message": message,
    }


def list_invited_users() -> list[dict]:
    return _list_users()


def revoke_invited_user(email: str) -> None:
    email = _validate_email(email)
    if email in _admin_emails():
        raise ValueError("Administrator access cannot be revoked here.")
    _revoke_user(email)


def render_invite_page(user: AccessUser) -> None:
    st.title("👥 Invite Friend")
    st.caption(
        "Approve a friend's sign-in email and optionally email them the private website link."
    )

    status = invite_system_status()

    if not status["persistent_storage"]:
        st.error("Invite storage is not configured yet.")
        st.info(
            "The one-click invite page needs Supabase credentials in Streamlit Secrets so "
            "approved emails survive app restarts and redeploys."
        )
        return

    if status["email_delivery"]:
        st.success("Access approval and invite-email delivery are ready.")
    else:
        st.warning(
            "Access approval is ready, but automatic invite emails are not configured. "
            "You can still approve a friend and copy the website link for them."
        )

    with st.form("invite_friend_page_form", clear_on_submit=True):
        invite_email = st.text_input(
            "Friend's email",
            placeholder="friend@example.com",
            autocomplete="email",
        )
        invite_submitted = st.form_submit_button(
            "Approve & invite",
            type="primary",
            use_container_width=True,
        )

    if invite_submitted:
        try:
            result = invite_user(invite_email, user.email)
            if result["email_sent"]:
                st.success(
                    f"{result['email']} is approved and the invitation email was sent."
                )
            else:
                st.success(f"{result['email']} is approved for access.")
                st.caption(result["message"])
                if status["app_url"]:
                    st.code(status["app_url"], language=None)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
        except requests.RequestException as exc:
            st.error(f"The invite could not be completed: {exc}")

    st.markdown("### Approved users")
    try:
        users = list_invited_users()
    except requests.RequestException as exc:
        st.error(f"Could not load the access list: {exc}")
        users = []

    if not users:
        st.info("No invited users yet.")
        return

    active_users = [u for u in users if bool(u.get("active"))]
    st.dataframe(
        [
            {
                "Email": u.get("email", ""),
                "Active": bool(u.get("active")),
                "Invited by": u.get("invited_by", ""),
                "Invited at": u.get("invited_at", ""),
            }
            for u in users
        ],
        hide_index=True,
        use_container_width=True,
    )

    revokable = sorted(
        {
            _normalise_email(u.get("email", ""))
            for u in active_users
            if _normalise_email(u.get("email", ""))
            and _normalise_email(u.get("email", "")) not in _admin_emails()
        }
    )

    if revokable:
        with st.form("revoke_friend_page_form"):
            revoke_email = st.selectbox("Revoke access", revokable)
            revoke_submitted = st.form_submit_button(
                "Revoke selected user",
                use_container_width=True,
            )

        if revoke_submitted:
            try:
                revoke_invited_user(revoke_email)
                st.success(f"Access revoked for {revoke_email}.")
                st.rerun()
            except (ValueError, RuntimeError, requests.RequestException) as exc:
                st.error(f"Could not revoke access: {exc}")


def _render_admin_panel(user: AccessUser) -> None:
    with st.expander("Admin access", expanded=False):
        st.caption("Approve an email, send its invitation, or revoke an existing user.")

        if not _database_is_configured():
            st.warning(
                "Persistent invites are not configured yet. Add the Supabase settings "
                "from README.md to Streamlit Secrets."
            )
            return

        with st.form("invite_user_form", clear_on_submit=True):
            invite_email = st.text_input(
                "Invite email",
                placeholder="person@example.com",
                autocomplete="email",
            )
            invite_submitted = st.form_submit_button(
                "Approve & send invite",
                type="primary",
                use_container_width=True,
            )

        if invite_submitted:
            try:
                email = _validate_email(invite_email)
                if email in _admin_emails():
                    st.info("That email is already an administrator.")
                else:
                    _approve_user(email, user.email)
                    sent, message = _send_invite_email(email, user.email)
                    if sent:
                        st.success(f"{email} was approved. {message}")
                    else:
                        st.warning(f"{email} was approved. {message}")
                        app_url = str(_access_config().get("app_url", "")).strip()
                        if app_url:
                            st.code(app_url, language=None)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
            except requests.RequestException as exc:
                st.error(f"The invite could not be completed: {exc}")

        try:
            users = _list_users()
        except requests.RequestException as exc:
            st.error(f"Could not load the access list: {exc}")
            users = []

        if users:
            active_users = [u for u in users if bool(u.get("active"))]
            st.caption(f"{len(active_users)} active invited user(s)")
            st.dataframe(
                [
                    {
                        "Email": u.get("email", ""),
                        "Active": bool(u.get("active")),
                        "Invited by": u.get("invited_by", ""),
                        "Invited at": u.get("invited_at", ""),
                    }
                    for u in users
                ],
                hide_index=True,
                use_container_width=True,
            )

            revokable = sorted(
                {
                    _normalise_email(u.get("email", ""))
                    for u in active_users
                    if _normalise_email(u.get("email", ""))
                    and _normalise_email(u.get("email", "")) not in _admin_emails()
                }
            )
            if revokable:
                with st.form("revoke_user_form"):
                    revoke_email = st.selectbox("Revoke access", revokable)
                    revoke_submitted = st.form_submit_button(
                        "Revoke selected user",
                        use_container_width=True,
                    )
                if revoke_submitted:
                    try:
                        _revoke_user(revoke_email)
                        st.success(f"Access revoked for {revoke_email}.")
                        st.rerun()
                    except (RuntimeError, requests.RequestException) as exc:
                        st.error(f"Could not revoke access: {exc}")


def render_access_sidebar(user: AccessUser) -> str | None:
    action = None
    with st.sidebar:
        st.subheader("Account")
        st.caption(f"Signed in as {user.email}")
        if user.is_admin:
            st.caption("Administrator")
            if st.button(
                "👥 Invite Friend",
                use_container_width=True,
                key="open_invite_friend_page",
            ):
                st.session_state["admin_page"] = "invite_friend"

        st.button("Log out", on_click=st.logout, use_container_width=True)

        if st.session_state.get("admin_page") == "invite_friend":
            action = "invite_friend"

    return action
