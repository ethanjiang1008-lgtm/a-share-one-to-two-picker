#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from market_data_sina import fetch_kline

ROOT=Path(__file__).resolve().parents[1]
REPORT_DIR=ROOT/"reports"/"daily"
EVAL_DIR=ROOT/"reports"/"evaluation"
RECORDS=EVAL_DIR/"daily_topk_validation.csv"
SUMMARY=EVAL_DIR/"topk_summary.json"
TOPKS=(1,3,6,10)

def load(p):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return None

def f(x, default=0.0):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (TypeError,ValueError):
        return default

def actual_metrics(code, prediction_row, verify_date):
    """Read the verified next-trading-day daily bar for one predicted stock."""
    try:
        bars=fetch_kline(code,80)
        bar=next((x for x in bars if str(x.get("date",""))==verify_date),None)
        if not bar:
            return None
        prev_close=f(prediction_row.get("price"))
        o,h,c=f(bar.get("open")),f(bar.get("high")),f(bar.get("close"))
        if prev_close<=0:
            return None
        return {
            "next_day_open_return":(o/prev_close-1) if o else None,
            "next_day_high_return":(h/prev_close-1) if h else None,
            "next_day_close_return":(c/prev_close-1) if c else None,
            "next_day_close":c,
        }
    except Exception:
        return None

def main():
    files=sorted(REPORT_DIR.glob("*_one_to_two_v1.json"))
    reports=[(p,load(p)) for p in files]
    reports=[x for x in reports if x[1] and x[1].get("rows")]
    if not reports:
        print("No daily reports for rolling TopK validation.")
        return 0

    # 只有收盘后的完整日K结果才能作为“已经发生”的下一交易日验证。
    # 盘中运行只是预测快照，不能拿未收盘的二板名单做验证。
    _,current=reports[-1]
    if current.get("market_data_mode") != "daily_close":
        print(f"Latest report is {current.get('market_data_mode')}; skip TopK validation until daily_close.")
        return 0
    verify_date=current.get("analysis_date") or current.get("date")
    actual_2plus={str(x) for x in (current.get("two_plus_codes") or [])}
    if not actual_2plus and current.get("two_plus_count",0):
        print("Current report lacks two_plus_codes; skip validation.")
        return 0

    # Find any prediction made specifically for this realized trading date.
    candidates=[d for _,d in reports[:-1] if d.get("prediction_date")==verify_date and d.get("rows")]
    if not candidates:
        print(f"No prior prediction found for verification date {verify_date}.")
        return 0

    EVAL_DIR.mkdir(parents=True,exist_ok=True)
    existing={}
    if RECORDS.exists():
        with RECORDS.open(encoding="utf-8-sig",newline="") as fh:
            for r in csv.DictReader(fh):
                existing[(r.get("prediction_date",""),r.get("code",""),r.get("rank",""))]=r

    jobs=[]
    for pred in candidates:
        for r in sorted(pred.get("rows",[]),key=lambda x:int(x.get("rank",999)))[:10]:
            jobs.append((pred,r))

    realized={}
    with ThreadPoolExecutor(max_workers=10) as ex:
        fs={ex.submit(actual_metrics,str(r.get("code","")),r,verify_date):(pred,r) for pred,r in jobs}
        for fut in as_completed(fs):
            pred,r=fs[fut]
            key=(pred.get("prediction_date",""),str(r.get("code","")),str(int(r.get("rank",999))))
            realized[key]=fut.result()

    for pred,r in jobs:
        pdate=pred.get("prediction_date")
        rank=int(r.get("rank",999))
        code=str(r.get("code",""))
        key=(pdate,code,str(rank))
        m=realized.get(key) or {}
        is_2=1 if code in actual_2plus else 0
        conclusion="晋级二板" if is_2 else "未晋级二板"
        if m.get("next_day_close_return") is not None:
            close_pct=f(m["next_day_close_return"])*100
            conclusion += f"；次日收盘相对首板收盘 {close_pct:+.2f}%"
        existing[key]={
            "prediction_date":pdate,
            "verification_date":verify_date,
            "rank":rank,
            "code":code,
            "name":r.get("name",""),
            "score":r.get("score",0),
            "is_2board":is_2,
            "next_day_open_return":m.get("next_day_open_return"),
            "next_day_high_return":m.get("next_day_high_return"),
            "next_day_close_return":m.get("next_day_close_return"),
            "next_day_close":m.get("next_day_close"),
            "review_status":"verified",
            "review_conclusion":conclusion,
        }

    rows=sorted(existing.values(),key=lambda r:(r.get("prediction_date",""),int(r.get("rank",999))))
    fields=["prediction_date","verification_date","rank","code","name","score","is_2board",
            "next_day_open_return","next_day_high_return","next_day_close_return","next_day_close",
            "review_status","review_conclusion"]
    with RECORDS.open("w",encoding="utf-8-sig",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)

    by_date=defaultdict(list)
    for r in rows: by_date[r["prediction_date"]].append(r)

    # Build the true same-day baseline from the full prior prediction universe.
    # This is independent of TopK selection and therefore provides the correct
    # denominator for lift calculations.
    baseline_by_date={}
    for pred in candidates:
        pdate=pred.get("prediction_date","")
        all_rows=pred.get("rows",[])
        if not pdate or not all_rows:
            continue
        baseline_by_date[pdate]={
            "selected":len(all_rows),
            "hits":sum(1 for item in all_rows if str(item.get("code","")) in actual_2plus)
        }
    metrics={}
    for k in TOPKS:
        selected=hits=days=day_hits=0
        for _,rs in by_date.items():
            top=[x for x in rs if int(x["rank"])<=k and x.get("review_status")=="verified"]
            if not top: continue
            days+=1
            selected+=len(top)
            h=sum(int(x["is_2board"]) for x in top)
            hits+=h
            day_hits+=int(h>0)
        baseline_selected=sum(v["selected"] for v in baseline_by_date.values())
        baseline_hits=sum(v["hits"] for v in baseline_by_date.values())
        baseline_rate=baseline_hits/baseline_selected if baseline_selected else None
        precision=hits/selected if selected else None
        metrics[str(k)]={
            "k":k,"days":days,"selected":selected,"hits":hits,
            "precision":precision,
            "baseline_rate":baseline_rate,
            "lift":(precision/baseline_rate) if precision is not None and baseline_rate else None,
            "day_hit_days":day_hits,"day_hit_rate":day_hits/days if days else None
        }

    summary={
        "model_version":current.get("model_version"),
        "source":"rolling_live_validation",
        "notes":"只使用已经发生的下一交易日结果；不修改V1模型参数。",
        "metrics":metrics
    }
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0

if __name__=="__main__": raise SystemExit(main())
