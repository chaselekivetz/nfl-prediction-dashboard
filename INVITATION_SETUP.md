# Password invitations

Password invitation setup for the Gridiron Central dashboard.

The administrator enters an email and clicks Approve & invite. Auth0 creates a database account when needed and sends its Change Password email. The recipient chooses a password, and Auth0 verifies the email after that flow. Existing accounts retain their password until they use the invitation. Users remain accepted until revoked; Restore access reactivates a revoked user without resetting their password.


## Production configuration


1. In Auth0, enable an email/password Database connection for the existing Streamlit application. Disable public signups on that connection. Ensure Universal Login offers that connection rather than forcing the passwordless email connection. Preserve the administrator's existing login method while testing.
2. Create an Auth0 Machine-to-Machine application authorized for the Management API with only `read:users` and `create:users` permissions.
3. Set the Streamlit application's Auth0 Application Login URI to the dashboard URL; preserve its `/oauth2callback` allowed callback. Password setup should offer a return to that application. Set a suitable password policy on the database connection.
4. Add the following section to private Streamlit Secrets, using actual values there only:


```toml
[invite_auth0]
domain = "YOUR-TENANT.us.auth0.com"
client_id = "YOUR-MACHINE-TO-MACHINE-CLIENT-ID"
client_secret = "YOUR-MACHINE-TO-MACHINE-SECRET"
connection = "Username-Password-Authentication"
```


5. Keep the existing `[auth]` login client and `[access]` Supabase configuration. `access.app_url` is optional and only used as a manual fallback display.
6. In Auth0, configure the tenant email provider and customize the **Change Password** template so the message clearly says it is a Gridiron Central invitation. Auth0’s development email provider may be limited; use SMTP for reliable production delivery.
7. Keep ordinary members out of `access.approved_emails`: that legacy static list bypasses database approval. The UI refuses to claim revocation succeeded for a static member. Administrator accounts remain protected.


## Acceptance check before rollout


Invite a non-admin test account; verify actual email receipt, password setup, verified email claim, and login. Check repeat invitation, expired ticket, wrong/unverified email, denied unapproved account, revoke while logged in, restoration, database outage, and email-provider failure. Revocation is checked on subsequent Streamlit reruns; an already rendered page cannot be recalled. An invitation ticket cannot bypass revoked database access. Email failures retain approval and allow an administrator to retry.


This deployment has been configured for Auth0 email delivery, but no test invitation has been sent by this session. Complete the live acceptance check before inviting real users.


Reference: https://auth0.com/docs/customize/email/send-email-invitations-for-application-signup
