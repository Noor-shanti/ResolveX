"""Create the M9 golden cases through the running ResolveX API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import GOLDEN_SCENARIOS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        response = requests.get(f"{base_url}/api/v1/engineer/tickets", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"Cannot reach ResolveX API at {base_url}: {exc}", file=sys.stderr)
        return 1

    existing = response.json()
    if existing:
        print(
            f"Refusing to seed {len(existing)} ticket(s): the engineer inbox is not empty. "
            "Review or clear it before retrying.",
            file=sys.stderr,
        )
        return 2

    created: list[dict[str, str]] = []
    manifest_path = ROOT / "data/processed/m9_ticket_manifest.json"

    def save_manifest(complete: bool) -> None:
        manifest_path.write_text(
            json.dumps(
                {
                    "base_url": base_url,
                    "complete": complete,
                    "ticket_count": len(created),
                    "tickets": created,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for scenario in GOLDEN_SCENARIOS:
        payload = {
            "title": scenario.title,
            "description": scenario.body,
            "product_type": scenario.product_type,
            "ubuntu_release": scenario.ubuntu_release,
        }
        try:
            result = requests.post(
                f"{base_url}/api/v1/tickets", json=payload, timeout=15
            )
            result.raise_for_status()
        except requests.RequestException as exc:
            print(
                f"Stopped at {scenario.scenario_id}; {len(created)} ticket(s) were created. "
                f"API error: {exc}",
                file=sys.stderr,
            )
            break
        ticket = result.json()
        created.append({
            "scenario_id": scenario.scenario_id,
            "ticket_id": ticket["ticket_id"],
        })
        save_manifest(complete=False)
        print(f"{scenario.scenario_id} -> {ticket['ticket_id']}")

    if len(created) != len(GOLDEN_SCENARIOS):
        return 1

    save_manifest(complete=True)
    print(f"Created {len(created)} golden tickets. Mapping saved to {manifest_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
