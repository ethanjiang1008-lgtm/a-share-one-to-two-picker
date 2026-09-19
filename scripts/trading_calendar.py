#!/usr/bin/env python3
"""A股交易日辅助。

日常运行优先使用固定的交易日历配置；周末与已知休市日自动跳过。
交易日历每年应根据上交所/深交所官方安排更新。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
CALENDAR_PATH = ROOT / "config" / "trading_calendar.json"


def _load_holidays() -> set[date]:
    try:
        payload = json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    except Exception:
        return set()
    holidays = set()
    for value in payload.get("holidays", []):
        try:
            holidays.add(date.fromisoformat(str(value)))
        except ValueError:
            continue
    return holidays


def is_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in _load_holidays()


def next_trading_day(day: date) -> date:
    candidate = day + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def previous_trading_day(day: date) -> date:
    candidate = day - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate
