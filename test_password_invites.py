import unittest
from unittest.mock import patch, Mock
from password_invites import setup_link


class InvitationTests(unittest.TestCase):
    def response(self, body):
        response = Mock()
        response.json.return_value = body
        return response

    @patch('password_invites.requests.get')
    @patch('password_invites.requests.post')
    def test_new_user_is_unverified_until_ticket_used(self, post, get):
        get.return_value = self.response([])
        post.side_effect = [self.response({'access_token': 'token'}),
                            self.response({'user_id': 'auth0|test'}),
                            self.response({'ticket': 'https://tenant.auth0.com/ticket'})]
        result = setup_link('friend@example.com', dict(domain='tenant.auth0.com',
            client_id='m2m', client_secret='secret', connection='database'), 'app')
        self.assertEqual(result, 'https://tenant.auth0.com/ticket')
        self.assertFalse(post.call_args_list[1].kwargs['json']['email_verified'])
        ticket = post.call_args_list[2].kwargs['json']
        self.assertTrue(ticket['mark_email_as_verified'])
        self.assertEqual(ticket['ttl_sec'], 86400)

    @patch('password_invites.requests.get')
    @patch('password_invites.requests.post')
    def test_existing_account_password_is_not_overwritten(self, post, get):
        get.return_value = self.response([{'user_id': 'auth0|existing',
            'identities': [{'connection': 'database'}]}])
        post.side_effect = [self.response({'access_token': 'token'}),
                            self.response({'ticket': 'https://tenant.auth0.com/ticket'})]
        setup_link('friend@example.com', dict(domain='tenant.auth0.com',
            client_id='m2m', client_secret='secret', connection='database'), 'app')
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs['json']['user_id'], 'auth0|existing')

    def test_invalid_domain(self):
        with self.assertRaises(ValueError):
            setup_link('a@example.com', {'domain': 'evil.test/path'}, 'app')


if __name__ == '__main__':
    unittest.main()
