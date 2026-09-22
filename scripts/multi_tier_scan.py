#!/usr/bin/env python3
from __future__ import annotations
import datetime,json,math,statistics
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

from market_data_sina import fetch_all_stocks,fetch_kline,is_main_board
from trading_calendar import next_trading_day

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"config/models_v2.json"
OUT=ROOT/"docs/data/multi_tier_latest.json"
LIMIT_UP=0.095; MIN_HISTORY=60; WORKERS=10; KLINE_COUNT=100

FEATURES=[
"ret_1d","ret_3d","ret_5d","ret_10d","ret_20d","volume_5_vs_20",
"close_above_ma20","ma5_gt_ma10","ma10_gt_ma20","near_high20_pct","above_low20_pct",
"up_days_5","volatility_10d","amplitude_1d","market_first_count","market_2plus_count",
"market_zt_count","prev_market_first_count","prev_market_2plus_count","prev_market_1to2_rate",
"open_gap_pct","open_gap_band_high","open_gap_band_mid","open_gap_band_low","open_to_close_pct",
"lower_wick_ratio","intraday_pullback_pct","open_position_in_day","amplitude_vs_20d",
"volume_vs_20d","near_limit_open","close_near_high","board_level","board_run_3d",
"board_run_5d","is_highest_board","is_second_highest_board"
]

def f(x,d=0.0):
    try:
        v=float(x); return v if math.isfinite(v) else d
    except (TypeError,ValueError): return d
def pct(a,b): return (a/b-1)*100 if b else 0.0
def limit_up(b,p):
    pc=f(p.get("close")); return pc>0 and f(b.get("close"))/pc-1>=LIMIT_UP

def board_level(bars,i):
    if i==0 or not limit_up(bars[i],bars[i-1]): return 0
    lv=1; j=i-1
    while j>0 and limit_up(bars[j],bars[j-1]):
        lv+=1; j-=1
    return lv

def calc_features(bars,i,level,cur,prev):
    pre=bars[:i+1]
    c=[f(x.get("close")) for x in pre]; v=[f(x.get("volume")) for x in pre]
    b,p=bars[i],bars[i-1]
    o,h,l,cl=f(b.get("open")),f(b.get("high")),f(b.get("low")),f(b.get("close")); pc=f(p.get("close"))
    ma5=statistics.fmean(c[-5:]); ma10=statistics.fmean(c[-10:]); ma20=statistics.fmean(c[-20:])
    v5=statistics.fmean(v[-5:]); v20=statistics.fmean(v[-20:])
    amps=[(f(x.get("high"))-f(x.get("low")))/f(x.get("close"))*100 for x in pre[-20:] if f(x.get("close"))>0]
    avg_amp=statistics.fmean(amps) if amps else 1.0
    rng=max(h-l,1e-9); gap=pct(o,pc)
    levels=sorted(set(cur.get("levels",{}).values()),reverse=True)
    highest=cur.get("max_board",level) or level
    second=levels[1] if len(levels)>1 else -1
    prev_first=len(prev.get("first",set()))
    prev_two=len(prev.get("two",set()))
    prev_rate=(len(prev.get("first",set()) & cur.get("two",set()))/prev_first) if prev_first else 0.16185320352130533
    return {
      "ret_1d":pct(cl,c[-2]),"ret_3d":pct(cl,c[-4]),"ret_5d":pct(cl,c[-6]),"ret_10d":pct(cl,c[-11]),"ret_20d":pct(cl,c[-21]),
      "volume_5_vs_20":v5/v20 if v20 else 1,"close_above_ma20":int(cl>ma20),"ma5_gt_ma10":int(ma5>ma10),"ma10_gt_ma20":int(ma10>ma20),
      "near_high20_pct":pct(cl,max(c[-20:])), "above_low20_pct":pct(cl,min(c[-20:])),
      "up_days_5":sum(c[z]>c[z-1] for z in range(len(c)-5,len(c))),"volatility_10d":statistics.pstdev(c[-10:])/cl*100,
      "amplitude_1d":(h-l)/cl*100,"market_first_count":len(cur["first"]),"market_2plus_count":len(cur["two"]),
      "market_zt_count":len(cur["zt"]),"prev_market_first_count":prev_first,"prev_market_2plus_count":prev_two,
      "prev_market_1to2_rate":prev_rate,"open_gap_pct":gap,"open_gap_band_high":int(gap>=8.5),
      "open_gap_band_mid":int(5<=gap<8.5),"open_gap_band_low":int(gap<5),"open_to_close_pct":pct(cl,o),
      "lower_wick_ratio":(o-l)/rng,"intraday_pullback_pct":pct(o,l),"open_position_in_day":(o-l)/rng,
      "amplitude_vs_20d":((h-l)/pc*100)/avg_amp if pc else 0,"volume_vs_20d":f(b.get("volume"))/v20 if v20 else 1,
      "near_limit_open":int(gap>=8),"close_near_high":int(h>0 and (h-cl)/h<=.002),"board_level":level,
      "board_run_3d":int(level>=3),"board_run_5d":int(level>=5),"is_highest_board":int(level==highest),
      "is_second_highest_board":int(level==second and level<highest)
    }

def score(feat,model):
    z=model["intercept"]
    for item in model["features"]:
        z += item["coef"]*((f(feat.get(item["name"]))-item["mean"])/(item["std"] or 1e-9))
    return 1/(1+math.exp(max(-35,min(35,-z))))

def main():
    now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
    analysis_date=now.date()
    prediction_date=next_trading_day(analysis_date).isoformat()
    universe=[x for x in fetch_all_stocks() if is_main_board(str(x.get("code","")),str(x.get("name","")))]
    stocks={}; failed=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fs={ex.submit(fetch_kline,str(x["code"]),KLINE_COUNT):x for x in universe}
        for fut in as_completed(fs):
            x=fs[fut]
            try:
                bars=sorted(fut.result(),key=lambda q:str(q.get("date","")))
                if len(bars)>=MIN_HISTORY:
                    stocks[str(x["code"])]=(str(x["name"]),bars)
            except Exception:
                failed+=1
    current={}
    for code,(name,bars) in stocks.items():
        idx=next((i for i,b in enumerate(bars) if str(b.get("date"))==analysis_date.isoformat()),None)
        if idx is None or idx==0 or not limit_up(bars[idx],bars[idx-1]): continue
        lv=board_level(bars,idx)
        current[code]={"name":name,"level":min(lv,6),"bars":bars,"i":idx}
    first={c for c,x in current.items() if x["level"]==1}
    two={c for c,x in current.items() if x["level"]>=2}
    zt=set(current)
    levels={c:x["level"] for c,x in current.items()}
    cur={"first":first,"two":two,"zt":zt,"levels":levels,"max_board":max(levels.values(),default=0)}
    prev={"first":set(),"two":set(),"zt":set(),"levels":{},"max_board":0}
    models=json.loads(MODEL_PATH.read_text(encoding="utf-8")) if MODEL_PATH.exists() else {}
    buckets={str(k):[] for k in range(1,7)}
    for code,x in current.items():
        lv=x["level"]; model=models.get(str(lv))
        if not model: continue
        feat=calc_features(x["bars"],x["i"],lv,cur,prev)
        buckets[str(lv)].append({
            "code":code,"name":x["name"],"level":lv,
            "price":f(x["bars"][x["i"]].get("close")),"score":score(feat,model),
            "prediction_date":prediction_date
        })
    for k in buckets:
        buckets[k].sort(key=lambda r:(-r["score"],r["code"]))
        buckets[k]=buckets[k][:10]
    payload={"status":"ok","analysis_date":analysis_date.isoformat(),"prediction_date":prediction_date,
             "model_version":"multi-tier-v2","universe":"沪深主板","failed":failed,
             "market":{"first_count":len(first),"two_plus_count":len(two),"zt_count":len(zt),"max_board":cur["max_board"]},
             "levels":buckets}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__": raise SystemExit(main())
