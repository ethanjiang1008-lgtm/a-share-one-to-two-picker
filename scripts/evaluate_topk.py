#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT_DIR=ROOT/"reports"/"daily"
EVAL_DIR=ROOT/"reports"/"evaluation"
RECORDS=EVAL_DIR/"daily_topk_validation.csv"
SUMMARY=EVAL_DIR/"topk_summary.json"
TOPKS=(1,3,6,10)

def load(p):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return None

def main():
    files=sorted(REPORT_DIR.glob("*_one_to_two_v1.json"))
    if len(files)<2:
        print("Not enough daily reports for rolling TopK validation."); return 0
    reports=[(p,load(p)) for p in files]; reports=[x for x in reports if x[1]]
    _,current=reports[-1]
    verify_date=current.get("analysis_date") or current.get("date")
    actual={str(x) for x in (current.get("two_plus_codes") or [])}
    if not actual and current.get("two_plus_count",0):
        print("Current report lacks two_plus_codes; skip validation."); return 0
    candidates=[d for _,d in reports[:-1] if d.get("prediction_date")==verify_date and d.get("rows")]
    if not candidates:
        print(f"No prior prediction found for verification date {verify_date}."); return 0

    EVAL_DIR.mkdir(parents=True,exist_ok=True); existing={}
    if RECORDS.exists():
        with RECORDS.open(encoding="utf-8-sig",newline="") as f:
            for r in csv.DictReader(f): existing[(r["prediction_date"],r["code"],r["rank"])]=r
    for pred in candidates:
        for r in sorted(pred.get("rows",[]),key=lambda x:int(x.get("rank",999)))[:10]:
            rank=int(r.get("rank",999)); code=str(r.get("code","")); pdate=pred.get("prediction_date")
            existing[(pdate,code,str(rank))]={"prediction_date":pdate,"verification_date":verify_date,"rank":rank,"code":code,"name":r.get("name",""),"score":r.get("score",0),"is_2board":1 if code in actual else 0}
    rows=sorted(existing.values(),key=lambda r:(r["prediction_date"],int(r["rank"])))
    with RECORDS.open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["prediction_date","verification_date","rank","code","name","score","is_2board"]); w.writeheader(); w.writerows(rows)

    by_date=defaultdict(list)
    for r in rows: by_date[r["prediction_date"]].append(r)
    metrics={}
    for k in TOPKS:
        selected=hits=days=day_hits=0
        for _,rs in by_date.items():
            top=[x for x in rs if int(x["rank"])<=k]
            if not top: continue
            days+=1; selected+=len(top); h=sum(int(x["is_2board"]) for x in top); hits+=h; day_hits+=int(h>0)
        metrics[str(k)]={"k":k,"days":days,"selected":selected,"hits":hits,"precision":hits/selected if selected else None,"day_hit_days":day_hits,"day_hit_rate":day_hits/days if days else None}
    summary={"model_version":current.get("model_version"),"source":"rolling_live_validation","notes":"只使用已经发生的下一交易日结果；不改变V1模型参数。","metrics":metrics}
    SUMMARY.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
