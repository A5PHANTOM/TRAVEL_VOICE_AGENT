import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
sys.path.insert(1, os.path.join(project_root, "pipecat/src"))

database_stub = ModuleType("app.database")

async def _save_interest_stub(*args, **kwargs):
    return 1


async def _find_open_lead_stub(*args, **kwargs):
    return None


async def _update_lead_stub(*args, **kwargs):
    return 2


database_stub.save_interest = _save_interest_stub
database_stub.find_open_lead = _find_open_lead_stub
database_stub.update_lead = _update_lead_stub
sys.modules.setdefault("app.database", database_stub)

from app.functions import register_interest


class TestRegisterInterest(unittest.IsolatedAsyncioTestCase):
    async def test_placeholder_destination_is_rejected(self):
        callback = AsyncMock()
        params = SimpleNamespace(result_callback=callback)

        await register_interest(
            params,
            destination="yes",
            lead_name="Edward",
            lead_email="edward@example.com",
            duration_days=5,
            accommodation="luxury",
            flight_needed=True,
        )

        callback.assert_awaited_once()
        result = callback.await_args.args[0]
        self.assertEqual(result["status"], "needs_destination")

    @patch("app.functions.save_interest", new_callable=AsyncMock)
    async def test_valid_lead_is_saved(self, mock_save_interest):
        callback = AsyncMock()
        params = SimpleNamespace(result_callback=callback)
        mock_save_interest.return_value = 42

        await register_interest(
            params,
            destination="Goa",
            lead_name="Edward",
            lead_email="edward@example.com",
            duration_days=5,
            accommodation="luxury",
            flight_needed=True,
        )

        callback.assert_awaited_once()
        result = callback.await_args.args[0]
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["id"], 42)
        mock_save_interest.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
