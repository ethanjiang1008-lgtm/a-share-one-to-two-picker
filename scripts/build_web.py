#!/usr/bin/env python3
from __future__ import annotations
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "daily"
WEB_DATA = ROOT / "docs" / "data" / "latest.json"

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def latest_report():
    files = sorted(REPORT_DIR.glob("*_one_to_two_v1.json"))
    return files[-1] if files else None

def _date_value(data: dict) -> date | None:
    raw = str(data.get("analysis_date") or data.get("date") or "")
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None

def main():
    src = latest_report()
    current = load_json(WEB_DATA) if WEB_DATA.exists() else None

    # live_scan.py 直接产生的 no_first_board / pre_open 状态也是有效运行结果。
    # 不能因为没有生成日报文件，就在这里偷偷恢复成旧日报，否则网页会出现
    # “工作流成功，但日期/预测完全不变”的假象。
    if current and current.get("status") in ("no_first_board", "pre_open"):
        print(
            f"Preserve live runtime status: status={current.get('status')}, "
            f"analysis_date={current.get('analysis_date')}, "
            f"prediction_date={current.get('prediction_date')}"
        )
        return 0

    if not src:
        print("No daily JSON report found; keep existing web data.")
        return 0

    data = load_json(src)
    src_date = _date_value(data)
    current_date = _date_value(current or {})

    # 只允许新日报覆盖网页，避免任何异常运行把网页回退到更旧的交易日。
    if current_date and src_date and src_date < current_date:
        print(
            f"Skip stale web overwrite: existing={current_date.isoformat()}, "
            f"candidate={src_date.isoformat()}"
        )
        return 0

    WEB_DATA.parent.mkdir(parents=True, exist_ok=True)
    WEB_DATA.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Web data updated: {WEB_DATA}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
