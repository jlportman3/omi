"""
Tests for POST /v1/sync/force — the server-initiated WAL drain trigger.

Closes the BLE-event-driven trigger gap from omi pain-point cluster #2:
when a user has a multi-hour WAL backlog sitting on the pendant's flash
because the phone's only drain trigger is BLE-reconnect, this endpoint
sends a high-priority silent FCM push (type=trigger_wal_drain) asking the
phone to ask the pendant to start draining now.

Pendant flash is NEVER touched by this code path — backend cannot reach
the pendant directly; only the phone can, via BLE. The push is a
*request*, not a mutation.
"""

import os
import unittest
from unittest.mock import patch


# ---------------------------------------------------------------------------
# 1. Structural tests — verify endpoint and helper exist with correct shape
# ---------------------------------------------------------------------------


class TestForceWalSyncStructure(unittest.TestCase):
    """Verify the endpoint and helper code exist with the expected contract."""

    @staticmethod
    def _read(relpath):
        path = os.path.join(os.path.dirname(__file__), '..', '..', *relpath.split('/'))
        with open(path) as f:
            return f.read()

    def test_endpoint_registered(self):
        source = self._read('routers/sync.py')
        self.assertIn('"/v1/sync/force"', source)
        self.assertIn('async def force_wal_sync', source)

    def test_endpoint_uses_user_auth(self):
        """Must require the authenticated user (no anonymous trigger)."""
        source = self._read('routers/sync.py')
        # The endpoint must use the standard user-uid auth dependency, so the
        # caller can only trigger a drain for themselves (or ADMIN_KEY<uid>
        # impersonation, per the env-driven admin path).
        start = source.index('async def force_wal_sync')
        end = source.find('\n@router.', start + 1)
        body = source[start:end] if end != -1 else source[start:]
        self.assertIn('Depends(auth.get_current_user_uid)', body)

    def test_endpoint_does_not_touch_pendant_or_files(self):
        """The endpoint must not call any filesystem or pendant-mutating API.

        The only operation is dispatching an FCM push; we explicitly forbid
        any of the destructive sync.py helpers in the endpoint body.
        """
        source = self._read('routers/sync.py')
        start = source.index('async def force_wal_sync')
        end = source.find('\n@router.', start + 1)
        body = source[start:end] if end != -1 else source[start:]
        for forbidden in (
            'shutil.rmtree',
            'os.remove',
            'os.unlink',
            'open(',
            'process_conversation',
            'upload_audio_chunk',
            'upload_sdcard_audio',
        ):
            self.assertNotIn(forbidden, body, f"force_wal_sync must not call {forbidden}")

    def test_helper_uses_data_only_high_priority_background_push(self):
        """The FCM helper must use the silent/background, high-priority path."""
        source = self._read('utils/notifications.py')
        self.assertIn('def send_force_wal_sync_message', source)
        start = source.index('def send_force_wal_sync_message')
        end = source.find('\ndef ', start + 1)
        body = source[start:end] if end != -1 else source[start:]
        # data-only (no `notification=` kwarg) + silent (`is_background=True`)
        # + high priority so APNs+FCM deliver promptly even when app is
        # backgrounded — but Android will only wake if the app process is
        # alive (cluster #2's remaining-killed-process risk is app-side).
        self.assertIn("'type': 'trigger_wal_drain'", body)
        self.assertIn('is_background=True', body)
        self.assertIn("priority='high'", body)


# ---------------------------------------------------------------------------
# 2. Behavioural test — helper dispatches to FCM and returns the count
# ---------------------------------------------------------------------------


class TestForceWalSyncDispatch(unittest.TestCase):
    """The helper must reach _send_to_user with the expected payload shape."""

    def test_dispatch_records_type_reason_and_timestamp(self):
        from utils import notifications

        with patch.object(notifications, '_send_to_user', return_value=3) as send_mock:
            count = notifications.send_force_wal_sync_message('uid-test', reason='admin_unblock')

        self.assertEqual(count, 3)
        self.assertEqual(send_mock.call_count, 1)
        # _send_to_user(user_id, tag, data=..., is_background=..., priority=...)
        _args, kwargs = send_mock.call_args
        self.assertTrue(kwargs.get('is_background'))
        self.assertEqual(kwargs.get('priority'), 'high')
        data = kwargs.get('data')
        self.assertEqual(data.get('type'), 'trigger_wal_drain')
        self.assertEqual(data.get('reason'), 'admin_unblock')
        self.assertTrue(data.get('requested_at'))
        # FCM data values must all be strings
        for k, v in data.items():
            self.assertIsInstance(v, str, f"{k}={v!r} is not a string")

    def test_long_reason_is_truncated(self):
        from utils import notifications

        long_reason = 'x' * 500
        with patch.object(notifications, '_send_to_user', return_value=1) as send_mock:
            notifications.send_force_wal_sync_message('uid-test', reason=long_reason)
        data = send_mock.call_args.kwargs['data']
        # 64-char cap keeps payload well under FCM's 4KB ceiling.
        self.assertLessEqual(len(data['reason']), 64)


if __name__ == '__main__':
    unittest.main()
