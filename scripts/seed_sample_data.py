"""Seed sample papers into ResearchLens API from data/sample_papers.json."""

import json
from pathlib import Path
import requests

API_URL = "http://localhost:8000/api/v1"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_papers.json"


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Sample dataset not found at: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    papers = payload.get("papers", [])
    if not papers:
        raise ValueError("No papers found in sample dataset")

    response = requests.post(f"{API_URL}/ingest/batch", json={"papers": papers}, timeout=60)
    response.raise_for_status()

    result = response.json()
    print(f"Seed successful: indexed {result.get('count', 0)} papers")


if __name__ == "__main__":
    main()
