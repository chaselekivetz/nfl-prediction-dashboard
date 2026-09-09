"""Auth0-managed password invitations using the tenant email provider."""
import secrets
import requests


def send_password_invite(email, config, login_client_id):
    domain = config['domain'].strip()
    if any(c in domain for c in '/:@?#') or not domain:
        raise ValueError('Use an Auth0 tenant hostname.')
    base = 'https://' + domain
    response = requests.post(base + '/oauth/token', json={
        'grant_type': 'client_credentials', 'client_id': config['client_id'],
        'client_secret': config['client_secret'], 'audience': base + '/api/v2/',
    }, timeout=10)
    response.raise_for_status()
    headers = {'Authorization': 'Bearer ' + response.json()['access_token']}
    response = requests.get(base + '/api/v2/users-by-email', headers=headers,
                            params={'email': email}, timeout=10)
    response.raise_for_status()
    users = [u for u in response.json() if any(
        i.get('connection') == config['connection'] for i in u.get('identities', []))]
    if len(users) > 1:
        raise ValueError('Multiple matching accounts require administrator review.')
    if users:
        user_id = users[0]['user_id']
    else:
        response = requests.post(base + '/api/v2/users', headers=headers, json={
            'connection': config['connection'], 'email': email,
            'password': secrets.token_urlsafe(48) + 'aA1!',
            'email_verified': False, 'verify_email': False,
        }, timeout=10)
        response.raise_for_status()
        user_id = response.json()['user_id']
    # Auth0 sends its configured Change Password email and marks the email
    # verified after the recipient completes the password setup flow.
    response = requests.post(base + '/dbconnections/change_password', json={
        'client_id': login_client_id, 'email': email,
        'connection': config['connection'],
    }, timeout=10)
    response.raise_for_status()
    return True
