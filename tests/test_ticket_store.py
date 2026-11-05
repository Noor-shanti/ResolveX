import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.ticket_store import TicketStore


class TicketStoreMigrationTests(unittest.TestCase):
    def test_adds_optional_environment_columns_without_replacing_existing_tickets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "legacy.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute("""
                    CREATE TABLE tickets (
                        ticket_id TEXT PRIMARY KEY,
                        tracking_token_hash TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT NOT NULL,
                        status TEXT NOT NULL,
                        status_note TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        analysis_json TEXT
                    )
                """)
                connection.execute("""
                    INSERT INTO tickets (
                        ticket_id, tracking_token_hash, title, description,
                        status, created_at, updated_at, analysis_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    "RX-LEGACY", "existing-hash", "Existing ticket", "Existing details",
                    "received", "2026-10-01T00:00:00+00:00", "2026-10-01T00:00:00+00:00", None,
                ))
                connection.commit()
            finally:
                connection.close()

            store = TicketStore(database_path)
            old_ticket = store.get_for_engineer("RX-LEGACY")
            created, _ = store.create(
                "New server ticket", "New details", "ubuntu_server", "24.04"
            )
            new_ticket = store.get_for_engineer(created["ticket_id"])

        self.assertEqual(old_ticket["title"], "Existing ticket")
        self.assertEqual(old_ticket["description"], "Existing details")
        self.assertIsNone(old_ticket["product_type"])
        self.assertIsNone(old_ticket["ubuntu_release"])
        self.assertEqual(new_ticket["product_type"], "ubuntu_server")
        self.assertEqual(new_ticket["ubuntu_release"], "24.04")


class TicketAgentConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = TicketStore(Path(self.temp_dir.name) / "tickets.sqlite3")
        self.ticket, _ = self.store.create("APT issue", "The package upgrade failed.")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_persists_ordered_engineer_and_agent_messages_with_agent_analysis(self):
        question = self.store.add_agent_message(
            self.ticket["ticket_id"], "engineer", "What evidence supports this?"
        )
        analysis = {
            "answer": "The official manual describes apt-get behavior.",
            "answer_citations": [{"source_id": "OD-1", "quote": "apt-get"}],
        }
        reply = self.store.add_agent_message(
            self.ticket["ticket_id"], "agent", analysis["answer"], analysis
        )

        messages = self.store.get_agent_conversation(self.ticket["ticket_id"])

        self.assertEqual([message["role"] for message in messages], ["engineer", "agent"])
        self.assertEqual(messages[0]["content"], question["content"])
        self.assertEqual(messages[1]["analysis"], analysis)
        self.assertEqual(reply["ticket_id"], self.ticket["ticket_id"])

    def test_rejects_empty_or_unsupported_messages_and_missing_tickets(self):
        with self.assertRaises(ValueError):
            self.store.add_agent_message(self.ticket["ticket_id"], "requester", "Hi")
        with self.assertRaises(ValueError):
            self.store.add_agent_message(self.ticket["ticket_id"], "engineer", "  ")
        with self.assertRaises(KeyError):
            self.store.add_agent_message("RX-MISSING", "engineer", "Question")
        self.assertIsNone(self.store.get_agent_conversation("RX-MISSING"))


if __name__ == "__main__":
    unittest.main()
