"""Load tests/golden/<stem>.json (written by oracle/<stem>.cpp against real CuTe)."""
import json
from pathlib import Path


def golden(stem: str):
    return json.loads((Path(__file__).parent / "golden" / f"{stem}.json").read_text())
