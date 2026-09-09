import unittest
from unittest.mock import patch, Mock
from password_invites import send_password_invite


class InvitationTests(unittest.TestCase):
    def response(self, body):
        response = Mock()
        response.json.return_value = body
        return response

    @patch('password_invites.requests.get')
    @patch('password_invites.requests.post')
    def test_new_user_receives_auth0_change_password_email(self, post, get):
        get.return_value = self.response([])
        post.side_effect = [self.response({'access_token': 'token'}),
                            self.response({'user_id': 'auth0|test'}),
                            self.response({'body': 'sent'})]
        result = send_password_invite('friend@example.com', dict(domain='tenant.auth0.com',
            client_id='m2m', client_secret='secret', connection='database'), 'app')
        self.assertTrue(result)
        self.assertFalse(post.call_args_list[1].kwargs['json']['email_verified'])
        self.assertEqual(post.call_args_list[2].args[0], 'https://tenant.auth0.com/dbconnections/change_password')
        self.assertEqual(post.call_args_list[2].kwargs['json']['email'], 'friend@example.com')

    @patch('password_invites.requests.get')
    @patch('password_invites.requests.post')
    def test_existing_account_password_is_not_overwritten(self, post, get):
        get.return_value = self.response([{'user_id': 'auth0|existing',
            'identities': [{'connection': 'database'}]}])
        post.side_effect = [self.response({'access_token': 'token'}),
                            self.response({'body': 'sent'})]
        send_password_invite('friend@example.com', dict(domain='tenant.auth0.com',
            client_id='m2m', client_secret='secret', connection='database'), 'app')
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args.kwargs['json']['email'], 'friend@example.com')
        self.assertEqual(post.call_args.args[0], 'https://tenant.auth0.com/dbconnections/change_password')

    def test_invalid_domain(self):
        with self.assertRaises(ValueError):
            send_password_invite('a@example.com', {'domain': 'evil.test/path'}, 'app')


if __name__ == '__main__':
    unittest.main()
