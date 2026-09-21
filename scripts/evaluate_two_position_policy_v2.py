#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"reports/backtest_recent_six_months.csv"
OUTPUT=ROOT/"reports/evaluation/two_position_policy_v2.json"

DEV_END="2026-07-08"
SCORE_GATE=0.40
GAP_CANDIDATES=[0.05,0.10,0.15,0.20,0.30,999.0]

def load_days():
    with INPUT.open(encoding="utf-8-sig",newline="") as fh:
        rows=list(csv.DictReader(fh))
    by=defaultdict(list)
    for r in rows:
        if r["window"]!="recent_6_months":
            continue
        by[(r["date"],r["prediction_date"])].append(r)
    days=[]
    for (date,pred),rs in by.items():
        rs.sort(key=lambda x:int(x["rank"]))
        if len(rs)<2:
            continue
        days.append({
            "date":date,"prediction_date":pred,
            "s1":float(rs[0]["score"]),"h1":int(rs[0]["is_2board"]),
            "s2":float(rs[1]["score"]),"h2":int(rs[1]["is_2board"]),
        })
    return sorted(days,key=lambda x:x["date"])

def summarize(days,gap_max):
    positions=hits=trade_days=two_days=any_hit=0
    for d in days:
        if d["s1"]<SCORE_GATE:
            continue
        trade_days+=1
        positions+=1; hits+=d["h1"]
        second=(d["s2"]>=SCORE_GATE and (d["s1"]-d["s2"])<=gap_max)
        if second:
            two_days+=1
            positions+=1; hits+=d["h2"]
        if d["h1"] or (second and d["h2"]):
            any_hit+=1
    return {
        "days":len(days),"trade_days":trade_days,"two_position_days":two_days,
        "selected_positions":positions,"hits":hits,
        "position_precision":hits/positions if positions else None,
        "any_hit_on_trade_days":any_hit/trade_days if trade_days else None,
        "average_positions_per_day":positions/len(days) if days else None
    }

def pct(x):
    return round(x*100,2) if x is not None else None

def main():
    if not INPUT.exists():
        raise SystemExit(f"missing {INPUT}")
    days=load_days()
    dev=[d for d in days if d["date"]<=DEV_END]
    oos=[d for d in days if d["date"]>DEV_END]
    grid=[]
    for g in GAP_CANDIDATES:
        grid.append({"gap_max":g,"development":summarize(dev,g),"strict_oos":summarize(oos,g)})
    best_dev=max(grid,key=lambda x:(x["development"]["position_precision"] or -1,-x["gap_max"]))
    best_gap=best_dev["gap_max"]

    result={
        "research_version":"two-position-policy-v2-20260920",
        "v1_parameters_changed":False,
        "score_gate":SCORE_GATE,
        "development_period":{"end":DEV_END,"days":len(dev)},
        "strict_oos_period":{"start":"2026-07-09","days":len(oos)},
        "selection_rule":"Choose the gap_max with highest development-period position precision; ties choose the smaller gap. Apply the selected rule once to strict OOS.",
        "grid":grid,
        "selected_gap_max":best_gap,
        "selected_rule":f"Top1 score >= {SCORE_GATE:.2f}; add Top2 only when Top2 score >= {SCORE_GATE:.2f} and Top1-Top2 score gap <= {best_gap:.2f}",
        "result_strict_oos":best_dev["strict_oos"],
        "interpretation":"This is a decision-layer experiment only. V1 score parameters are unchanged; the score is not interpreted as a calibrated probability."
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
        "selected_gap_max":best_gap,
        "development":best_dev["development"],
        "strict_oos":best_dev["strict_oos"],
        "baseline_no_gap":summarize(oos,999.0)
    },ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
