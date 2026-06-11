import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)

TEST_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_customer_memory.db"))
os.environ["DB_PATH"] = TEST_DB_PATH

from app.database import init_db, save_interest, get_leads
from app.customer_memory import (
    accommodation_to_budget_range,
    build_personalization_context,
    build_returning_greeting,
    lookup_customer_profile,
    record_interaction_from_lead,
    ensure_customer_memory_seeded,
)
from app.main import app


class TestCustomerMemory(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        await init_db(TEST_DB_PATH)

    async def asyncTearDown(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    async def test_record_and_lookup_customer(self):
        record = {
            "destination": "Thailand",
            "package_type": None,
            "duration_days": 4,
            "accommodation": "budget",
            "flight_needed": False,
            "lead_name": "Arjun",
            "lead_email": "m@gmail.com",
            "notes": None,
            "outgoing_number": "+919048310440",
        }
        lead_id = await save_interest("Thailand", record)
        customer_id = await record_interaction_from_lead(record, lead_id=lead_id, call_id="call-1")
        self.assertIsNotNone(customer_id)

        profile = await lookup_customer_profile(phone="+919048310440")
        self.assertIsNotNone(profile)
        customer = profile["customer"]
        self.assertEqual(customer["name"], "Arjun")
        self.assertEqual(customer["email"], "m@gmail.com")
        self.assertEqual(customer["preferred_accommodation"], "budget")
        self.assertEqual(customer["preferred_duration_days"], 4)
        self.assertFalse(customer["flight_preference"])
        self.assertEqual(customer["budget_range"], "Under ₹50,000")
        self.assertIn("Thailand", customer["destinations"])
        self.assertEqual(len(profile["interactions"]), 1)

    async def test_backfill_from_existing_leads(self):
        await save_interest(
            "Thailand",
            {
                "destination": "Thailand",
                "duration_days": 4,
                "accommodation": "budget",
                "flight_needed": False,
                "lead_name": "Arjun",
                "lead_email": "m@gmail.com",
                "outgoing_number": "+919048310440",
            },
        )
        count = await ensure_customer_memory_seeded()
        self.assertGreaterEqual(count, 1)

        profile = await lookup_customer_profile(email="m@gmail.com")
        self.assertIsNotNone(profile)
        self.assertEqual(profile["customer"]["name"], "Arjun")

    def test_personalization_helpers(self):
        profile = {
            "customer": {
                "name": "Arjun",
                "phone": "+919048310440",
                "email": "m@gmail.com",
                "destinations": ["Thailand"],
                "preferred_duration_days": 4,
                "preferred_accommodation": "budget",
                "budget_range": "Under ₹50,000",
                "flight_preference": False,
            },
            "interactions": [{"summary": "Thailand; 4 days; budget; no flights"}],
        }
        context = build_personalization_context(profile)
        self.assertIn("Arjun", context)
        self.assertIn("Thailand", context)
        self.assertIn("CUSTOMER MEMORY", context)

        greeting = build_returning_greeting(profile, "Default greeting")
        self.assertIn("Arjun", greeting)
        self.assertIn("Thailand", greeting)

    def test_budget_mapping(self):
        self.assertEqual(accommodation_to_budget_range("budget"), "Under ₹50,000")
        self.assertEqual(accommodation_to_budget_range("luxury"), "Above ₹1,50,000")

    def test_customer_api_endpoints(self):
        from fastapi.testclient import TestClient
        import asyncio

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.asyncSetUp())

        record = {
            "destination": "Goa",
            "duration_days": 3,
            "accommodation": "mid-range",
            "flight_needed": True,
            "lead_name": "Priya",
            "lead_email": "priya@example.com",
            "outgoing_number": "+919999999999",
        }
        loop.run_until_complete(save_interest("Goa", record))
        loop.run_until_complete(record_interaction_from_lead(record, lead_id=1))

        client = TestClient(app)
        response = client.get("/api/customers/lookup?phone=%2B919999999999")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["customer"]["name"], "Priya")
        self.assertEqual(data["customer"]["email"], "priya@example.com")

        list_response = client.get("/api/customers")
        self.assertEqual(list_response.status_code, 200)
        self.assertTrue(len(list_response.json()) >= 1)

        loop.run_until_complete(self.asyncTearDown())


if __name__ == "__main__":
    unittest.main()
