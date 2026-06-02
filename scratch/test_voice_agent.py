import os
import sys
import unittest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure the app and pipecat modules are in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
sys.path.insert(1, os.path.join(project_root, "pipecat/src"))

# Configure test database path
TEST_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_agents.db"))
os.environ["DB_PATH"] = TEST_DB_PATH

from app.database import init_db, save_transfer_context, get_transfer_context, save_interest, get_leads
from app.main import app

class TestDatabaseAndAPI(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Always remove any old test db and init
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        await init_db(TEST_DB_PATH)

    async def asyncTearDown(self):
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    async def test_database_transfer_context(self):
        # Test saving and retrieving transfer context
        call_id = "test-call-123"
        context = "Destination: Paris. Client Name: John Doe. Email: john@example.com."
        
        await save_transfer_context(call_id, context)
        retrieved = await get_transfer_context(call_id)
        
        self.assertEqual(retrieved, context)

        # Test retrieving non-existent call context
        non_existent = await get_transfer_context("invalid-id")
        self.assertIsNone(non_existent)

    async def test_database_leads(self):
        # Test saving interest
        lead_id = await save_interest("Paris", {"lead_name": "John Doe", "lead_email": "john@example.com"})
        self.assertIsNotNone(lead_id)
        
        leads = await get_leads()
        self.assertTrue(len(leads) > 0)
        self.assertEqual(leads[0]["package"], "Paris")
        self.assertEqual(leads[0]["lead"]["lead_name"], "John Doe")

    def test_twilio_whisper_endpoint(self):
        # Use FastAPI TestClient to test endpoints
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Insert a dummy context for the endpoint to retrieve
        call_id = "whisper-test-456"
        context = "Destination: Tokyo. Client Name: Jane Smith. Email: jane@example.com."
        
        # Run async function in loop to populate test database
        loop = asyncio.get_event_loop()
        loop.run_until_complete(save_transfer_context(call_id, context))

        # Test GET /twilio/whisper with call_id
        response = client.get(f"/twilio/whisper?call_id={call_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("application/xml", response.headers["content-type"])
        self.assertIn(context, response.text)
        self.assertIn("<Say>", response.text)

        # Test POST /twilio/whisper with call_id
        response_post = client.post(f"/twilio/whisper?call_id={call_id}")
        self.assertEqual(response_post.status_code, 200)
        self.assertIn("application/xml", response_post.headers["content-type"])
        self.assertIn(context, response_post.text)

        # Test GET /twilio/whisper with missing/invalid call_id (should fallback gracefully)
        response_fallback = client.get("/twilio/whisper?call_id=non-existent-id")
        self.assertEqual(response_fallback.status_code, 200)
        self.assertIn("No conversation summary available", response_fallback.text)

    def test_context_extraction(self):
        # We simulate context message matching the logic in app/main.py
        messages = [
            {"role": "user", "content": "Hi, I want to travel to Bali. My name is John Doe, email is john@example.org"},
            {"role": "assistant", "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "register_interest",
                        "arguments": '{"destination": "Bali", "lead_name": "John Doe"}'
                    }
                }
            ]}
        ]
        
        # Simulated extraction logic
        destination = "Not specified"
        email = "Not specified"
        name = "Not specified"
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tc in msg["tool_calls"]:
                    if tc.get("type") == "function" and tc.get("function", {}).get("name") == "register_interest":
                        try:
                            import json
                            args = json.loads(tc["function"]["arguments"])
                            if args.get("destination"):
                                destination = args.get("destination")
                            if args.get("lead_email"):
                                email = args.get("lead_email")
                            if args.get("lead_name"):
                                name = args.get("lead_name")
                        except Exception:
                            pass

        if email == "Not specified":
            import re
            email_regex = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
            for msg in messages:
                if msg.get("role") == "user" and msg.get("content"):
                    matches = email_regex.findall(msg.get("content"))
                    if matches:
                        email = matches[0]
                        break

        self.assertEqual(destination, "Bali")
        self.assertEqual(name, "John Doe")
        self.assertEqual(email, "john@example.org")

if __name__ == "__main__":
    unittest.main()
