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


if __name__ == "__main__":
    unittest.main()
