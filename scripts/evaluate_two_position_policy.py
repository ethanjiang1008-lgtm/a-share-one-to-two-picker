#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "reports" / "backtest_recent_six_months.csv"
OUTPUT = ROOT / "reports" / "evaluation" / "two_position_policy.json"

DEV_END = "2026-07-08"
TOP2_SCORE_GATE = 0.40
TOP1_SCORE_GATE = 0.40


def load_daily():
    with INPUT.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by = defaultdict(list)
    for r in rows:
        by[(r["window"], r["date"], r["prediction_date"])].append(r)

    out = []
    for (window, date, prediction_date), rs in by.items():
        rs.sort(key=lambda x: int(x["rank"]))
        if len(rs) < 2:
            continue
        out.append({
            "window": window,
            "date": date,
            "prediction_date": prediction_date,
            "score1": float(rs[0]["score"]),
            "hit1": int(rs[0]["is_2board"]),
            "score2": float(rs[1]["score"]),
            "hit2": int(rs[1]["is_2board"]),
        })
    return sorted(out, key=lambda x: (x["window"], x["date"]))


def summarize(days, selector):
    selected = hits = trade_days = both_days = any_hit_days = 0
    for d in days:
        picks = selector(d)
        if 1 in picks:
            trade_days += 1
        if len(picks) == 2:
            both_days += 1
        for k in picks:
            selected += 1
            hits += d["hit1"] if k == 1 else d["hit2"]
        if any((d["hit1"] if k == 1 else d["hit2"]) for k in picks):
            any_hit_days += 1
    return {
        "days": len(days),
        "trade_days": trade_days,
        "two_position_days": both_days,
        "selected_positions": selected,
        "hits": hits,
        "position_precision": hits / selected if selected else None,
        "day_any_hit_rate": any_hit_days / len(days) if days else None,
        "average_positions_per_trading_day": selected / len(days) if days else None,
    }


def metrics(days):
    return {
        "always_top1": summarize(days, lambda d: [1]),
        "always_top1_top2": summarize(days, lambda d: [1, 2]),
        "top1_gate_0.40_then_top2_gate_0.40": summarize(
            days,
            lambda d: ([1] if d["score1"] < TOP1_SCORE_GATE else ([1, 2] if d["score2"] >= TOP2_SCORE_GATE else [1]))
        ),
        "top1_only_when_score1_ge_0.40": summarize(
            days,
            lambda d: [1] if d["score1"] >= TOP1_SCORE_GATE else []
        ),
    }


def main():
    if not INPUT.exists():
        raise SystemExit(f"missing input: {INPUT}")
    daily = load_daily()
    recent = [d for d in daily if d["window"] == "recent_6_months"]
    dev = [d for d in recent if d["date"] <= DEV_END]
    oos = [d for d in daily if d["window"] == "strict_oos"]

    result = {
        "model_version": "one-to-two-v1-daily-proxy",
        "objective": "two-position decision layer; maximum two holdings per day",
        "model_parameters_changed": False,
        "top1_score_gate": TOP1_SCORE_GATE,
        "top2_score_gate": TOP2_SCORE_GATE,
        "development_period": {
            "end": DEV_END,
            "days": len(dev),
            "metrics": metrics(dev),
        },
        "strict_oos_period": {
            "start": "2026-07-09",
            "days": len(oos),
            "metrics": metrics(oos),
        },
        "interpretation": {
            "default_candidate": "Top1",
            "secondary_candidate": "Top2 only when its V1 score >= 0.40",
            "important": "The 0.40 gate is a research decision rule, not a calibrated probability threshold. Do not treat V1 scores as literal probabilities.",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
