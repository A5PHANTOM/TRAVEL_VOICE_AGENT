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

    def test_extract_details_from_history(self):
        from app.main import extract_details_from_history
        
        # Test Case 1: Standard conversation flow
        messages = [
            {"role": "assistant", "content": "Hi, I'm calling from Lifestyle Travels. Which destination are you planning to go to?"},
            {"role": "user", "content": "I'd like to go to Paris"},
            {"role": "assistant", "content": "Great, Paris! Who am I speaking with?"},
            {"role": "user", "content": "My name is John Doe"},
            {"role": "assistant", "content": "Thanks John. What is your email address?"},
            {"role": "user", "content": "john@example.com"},
            {"role": "assistant", "content": "Got it. How many days will you be travelling?"},
            {"role": "user", "content": "We will travel for 5 days"},
            {"role": "assistant", "content": "And what class of accommodation would you prefer: budget, mid-range, or luxury?"},
            {"role": "user", "content": "luxury please"},
            {"role": "assistant", "content": "Do you need flight requirements included?"},
            {"role": "user", "content": "yes"}
        ]
        
        details = extract_details_from_history(messages)
        self.assertEqual(details["destination"], "I'd like to go to Paris")
        self.assertEqual(details["lead_name"], "John Doe")
        self.assertEqual(details["lead_email"], "john@example.com")
        self.assertEqual(details["duration_days"], 5)
        self.assertEqual(details["accommodation"], "luxury please")
        self.assertEqual(details["flight_needed"], True)
        
        # Test Case 2: Confirmation pattern e.g. "Paris, is that correct?"
        messages_confirm = [
            {"role": "assistant", "content": "Hi, I'm calling from Lifestyle Travels. Which destination are you planning to go to?"},
            {"role": "user", "content": "Goa"},
            {"role": "assistant", "content": "Goa, is that correct?"},
            {"role": "user", "content": "Yes, correct"}
        ]
        details_confirm = extract_details_from_history(messages_confirm)
        self.assertEqual(details_confirm["destination"], "Goa")

    async def test_save_partial_lead_from_history(self):
        from app.main import save_partial_lead_from_history
        from app.database import get_leads
        
        messages = [
            {"role": "assistant", "content": "Hi, I'm calling from Lifestyle Travels. Which destination are you planning to go to?"},
            {"role": "user", "content": "Goa"},
            {"role": "assistant", "content": "Goa, is that correct?"},
            {"role": "user", "content": "Yes, correct"},
            {"role": "assistant", "content": "Who am I speaking with?"},
            {"role": "user", "content": "Alice Smith"}
        ]
        
        # Reset DB by dropping and re-initing first
        import os
        from app.database import init_db
        from scratch.test_voice_agent import TEST_DB_PATH
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        await init_db(TEST_DB_PATH)

        await save_partial_lead_from_history(messages, call_id="test_call_partial")
        
        leads = await get_leads()
        self.assertTrue(len(leads) > 0)
        lead = leads[0]
        self.assertEqual(lead["package"], "Goa")
        self.assertEqual(lead["lead"]["lead_name"], "Alice Smith")
        self.assertEqual(lead["lead"]["lead_email"], None)
        self.assertIn("test_call_partial", lead["lead"]["notes"])

        # Test duplicate prevention skip if register_interest successfully called
        messages_registered = messages + [
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "register_interest", "arguments": '{"destination": "Goa", "lead_name": "Alice Smith", "lead_email": "alice@example.com"}'}}]},
            {"role": "tool", "content": '{"status": "ok", "id": 123}'}
        ]
        
        initial_count = len(await get_leads())
        await save_partial_lead_from_history(messages_registered, call_id="test_call_registered")
        final_count = len(await get_leads())
        self.assertEqual(initial_count, final_count)

if __name__ == "__main__":
    unittest.main()
