#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "daily"
WEB_DATA = ROOT / "docs" / "data" / "latest.json"

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def latest_report():
    files = sorted(REPORT_DIR.glob("*_one_to_two_v1.json"))
    return files[-1] if files else None

def main():
    src = latest_report()
    if not src:
        print("No daily JSON report found; keep existing web data.")
        return 0
    data = load_json(src)
    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Web data updated: {WEB_DATA}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())