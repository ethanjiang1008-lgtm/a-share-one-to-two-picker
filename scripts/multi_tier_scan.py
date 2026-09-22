#!/usr/bin/env python3
"""Standalone V2 daily scanner.

This helper intentionally reuses the same feature construction and scoring
logic as scripts/multi_tier_v2_backtest.py, so manual scans cannot drift from
the V2 model definition.
"""
from __future__ import annotations

import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from market_data_sina import fetch_all_stocks, fetch_kline, is_main_board
from multi_tier_v2_backtest import (
    KLINE_COUNT,
    MIN_HISTORY,
    WORKERS,
    build_today_prediction,
    market_states,
)

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"config"/"models_v2.json"
OUT=ROOT/"docs"/"data"/"multi_tier_latest.json"

def main():
    universe={
        str(x["code"]):str(x["name"])
        for x in fetch_all_stocks()
        if is_main_board(str(x.get("code","")),str(x.get("name","")))
        and x.get("code")
    }

    stocks={}
    failed=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fs={ex.submit(fetch_kline,c,KLINE_COUNT):c for c in universe}
        for fut in as_completed(fs):
            code=fs[fut]
            try:
                bars=sorted(fut.result(),key=lambda x:str(x.get("date","")))
                if len(bars)>=MIN_HISTORY+22:
                    idx={str(b.get("date")):i for i,b in enumerate(bars)}
                    stocks[code]=(universe[code],idx,bars)
                else:
                    failed+=1
            except Exception:
                failed+=1

    dates=sorted({
        datetime.date.fromisoformat(str(b["date"]))
        for _,_,bars in stocks.values()
        for b in bars if b.get("date")
    })
    dates=[d for d in dates if d<=datetime.date.today()]
    market=market_states(stocks,dates)
    models=json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else {}
    payload=build_today_prediction(stocks,models,dates,market)

    if payload is None:
        payload={
            "status":"no_data",
            "analysis_date":max(dates).isoformat() if dates else None,
            "prediction_date":None,
            "model_version":"multi-tier-v2",
            "universe":"沪深主板",
            "failed":failed,
            "levels":{str(k):[] for k in range(1,7)}
        }
    else:
        payload["failed"]=failed

    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__":
    raise SystemExit(main())
