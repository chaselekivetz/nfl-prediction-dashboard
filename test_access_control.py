import sys
import unittest
from unittest.mock import Mock, patch
sys.modules.setdefault('streamlit', Mock())
import requests
import access_control as access


class AccessTests(unittest.TestCase):
    @patch.object(access, '_admin_emails', return_value=set())
    @patch.object(access, '_static_approved_emails', return_value=set())
    @patch.object(access, '_is_database_approved', side_effect=requests.ConnectionError())
    def test_database_failure_denies_access(self, *mocks):
        self.assertFalse(access._is_approved('friend@example.com'))

    @patch.object(access, '_require_admin', return_value='admin@example.com')
    @patch.object(access, '_admin_emails', return_value=set())
    @patch.object(access, '_approve_user')
    @patch.object(access, '_send_invite_email', side_effect=requests.ConnectionError())
    def test_email_failure_keeps_approval_and_reports_failure(self, send, approve, *mocks):
        result = access.invite_user('friend@example.com', 'spoof@example.com')
        self.assertTrue(result['approved'])
        self.assertFalse(result['email_sent'])
        approve.assert_called_once_with('friend@example.com', 'admin@example.com')

    @patch.object(access, '_require_admin', side_effect=ValueError('Administrator required'))
    @patch.object(access, '_approve_user')
    def test_nonadmin_cannot_invite(self, approve, guard):
        with self.assertRaises(ValueError):
            access.invite_user('friend@example.com', 'admin@example.com')
        approve.assert_not_called()

    @patch.object(access, '_require_admin')
    @patch.object(access, '_admin_emails', return_value=set())
    @patch.object(access, '_static_approved_emails', return_value={'friend@example.com'})
    @patch.object(access, '_revoke_user')
    def test_static_member_cannot_appear_revoked(self, revoke, *mocks):
        with self.assertRaises(ValueError):
            access.revoke_invited_user('friend@example.com')
        revoke.assert_not_called()
