#!/usr/bin/env python3
from __future__ import annotations
import csv, datetime, json, math, statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression

from market_data_sina import fetch_all_stocks, fetch_kline, is_main_board

ROOT=Path(__file__).resolve().parents[1]
OUT_JSON=ROOT/"reports"/"multi_tier_v2_backtest.json"
OUT_CSV=ROOT/"reports"/"multi_tier_v2_samples.csv"
MODEL_OUT=ROOT/"config"/"models_v2.json"

LIMIT_UP=0.095
MIN_HISTORY=60
KLINE_COUNT=240
WORKERS=10
TRAIN_END=datetime.date(2026,7,8)
OOS_START=datetime.date(2026,7,9)
LEVELS=(1,2,3,4,5,6)

FEATURES=[
"ret_1d","ret_3d","ret_5d","ret_10d","ret_20d","volume_5_vs_20",
"close_above_ma20","ma5_gt_ma10","ma10_gt_ma20","near_high20_pct","above_low20_pct",
"up_days_5","volatility_10d","amplitude_1d",
"market_first_count","market_2plus_count","market_zt_count",
"prev_market_first_count","prev_market_2plus_count","prev_market_1to2_rate",
"open_gap_pct","open_gap_band_high","open_gap_band_mid","open_gap_band_low",
"open_to_close_pct","lower_wick_ratio","intraday_pullback_pct",
"open_position_in_day","amplitude_vs_20d","volume_vs_20d","near_limit_open","close_near_high",
"board_level","board_run_3d","board_run_5d","is_highest_board","is_second_highest_board"
]

def f(x,d=0.0):
    try:
        v=float(x); return v if math.isfinite(v) else d
    except (TypeError,ValueError): return d

def pct(a,b): return (a/b-1)*100 if b else 0.0

def limit_up(bar,prev):
    pc=f(prev.get("close"))
    return pc>0 and f(bar.get("close"))/pc-1>=LIMIT_UP

def make_features(bars,i,market,day,board_level):
    if i<MIN_HISTORY or i<21 or i==0 or i>=len(bars):
        return None
    b,prev=bars[i],bars[i-1]
    c=[f(x.get("close")) for x in bars[:i+1]]
    v=[f(x.get("volume")) for x in bars[:i+1]]
    if f(b.get("close"))<=0 or f(prev.get("close"))<=0: return None
    ma5=statistics.fmean(c[-5:]); ma10=statistics.fmean(c[-10:]); ma20=statistics.fmean(c[-20:])
    v5=statistics.fmean(v[-5:]); v20=statistics.fmean(v[-20:])
    amps=[(f(x.get("high"))-f(x.get("low")))/f(x.get("close"))*100 for x in bars[max(0,i-19):i+1] if f(x.get("close"))>0]
    avg_amp20=statistics.fmean(amps) if amps else 1.0
    o,h,l,cl=f(b.get("open")),f(b.get("high")),f(b.get("low")),f(b.get("close"))
    pc=f(prev.get("close")); rng=max(h-l,1e-9); gap=pct(o,pc)
    prev_day=day-datetime.timedelta(days=1)
    # market passed in for exact previous actual trading day where available
    cur=market.get(day,{"first":set(),"two":set(),"zt":set()})
    pv=market.get(prev_day,{"first":set(),"two":set(),"zt":set()})
    # For weekends/holidays, the caller remaps day keys; this fallback only affects sparse axes.
    prev_first = len(pv["first"])
    prev_two = len(pv["two"])
    prev_rate = (len(pv["first"] & cur["two"])/prev_first) if prev_first else 0.16185320352130533
    cur_max=max((max(len(s["first"]),0) for s in market.values()),default=0)
    highest=max([len(cur["two"])] + [0])
    highest_level=max((s.get("max_board",0) for s in market.values() if s),default=board_level)
    levels=sorted((s.get("max_board",0) for s in market.values() if s), reverse=True)
    second_level=levels[1] if len(levels)>1 else highest_level
    return {
      "ret_1d":pct(cl,c[-2]),"ret_3d":pct(cl,c[-4]),"ret_5d":pct(cl,c[-6]),
      "ret_10d":pct(cl,c[-11]),"ret_20d":pct(cl,c[-21]),
      "volume_5_vs_20":v5/v20 if v20 else 1.0,
      "close_above_ma20":int(cl>ma20),"ma5_gt_ma10":int(ma5>ma10),"ma10_gt_ma20":int(ma10>ma20),
      "near_high20_pct":pct(cl,max(c[-20:])),"above_low20_pct":pct(cl,min(c[-20:])),
      "up_days_5":sum(c[z]>c[z-1] for z in range(len(c)-5,len(c))),
      "volatility_10d":statistics.pstdev(c[-10:])/cl*100,
      "amplitude_1d":(h-l)/cl*100,
      "market_first_count":len(cur["first"]),"market_2plus_count":len(cur["two"]),
      "market_zt_count":len(cur["zt"]),"prev_market_first_count":prev_first,
      "prev_market_2plus_count":prev_two,"prev_market_1to2_rate":prev_rate,
      "open_gap_pct":gap,"open_gap_band_high":int(gap>=8.5),
      "open_gap_band_mid":int(5<=gap<8.5),"open_gap_band_low":int(gap<5),
      "open_to_close_pct":pct(cl,o),"lower_wick_ratio":(o-l)/rng,
      "intraday_pullback_pct":pct(o,l),"open_position_in_day":(o-l)/rng,
      "amplitude_vs_20d":((h-l)/pc*100)/avg_amp20 if pc else 0,
      "volume_vs_20d":f(b.get("volume"))/v20 if v20 else 1,
      "near_limit_open":int(gap>=8.0),"close_near_high":int(h>0 and (h-cl)/h<=0.002),
      "board_level":board_level,
      "board_run_3d":int(board_level>=3),
      "board_run_5d":int(board_level>=5),
      "is_highest_board":int(board_level==highest_level),
      "is_second_highest_board":int(board_level==second_level and board_level<highest_level),
    }

def board_level_for_day(bars,i):
    if i==0 or not limit_up(bars[i],bars[i-1]): return 0
    level=1; j=i-1
    while j>0 and limit_up(bars[j],bars[j-1]):
        level+=1; j-=1
    return level

def market_states(stock_data,dates):
    out={}
    for d in dates:
        ds=d.isoformat(); first=set(); two=set(); zt=set(); levels={}
        for code,(name,idx,bars) in stock_data.items():
            i=idx.get(ds)
            if i is None or i==0 or not limit_up(bars[i],bars[i-1]): continue
            zt.add(code); lv=board_level_for_day(bars,i); levels[code]=lv
            if lv==1: first.add(code)
            else: two.add(code)
        out[d]={"first":first,"two":two,"zt":zt,"levels":levels,"max_board":max(levels.values(),default=0)}
    return out

def fit_one_level(samples,level):
    train=[s for s in samples if s["date"]<=TRAIN_END.isoformat()]
    oos=[s for s in samples if s["date"]>=OOS_START.isoformat()]
    if len(train)<50 or len({s["y"] for s in train})<2: return None
    X=[[s["features"].get(f,0) for f in FEATURES] for s in train]; y=[s["y"] for s in train]
    model=LogisticRegression(max_iter=2000,class_weight=None,solver="liblinear")
    model.fit(X,y)
    means=[]; stds=[]
    for j,name in enumerate(FEATURES):
        vals=[float(row[j]) for row in X]
        mean=statistics.fmean(vals); sd=statistics.pstdev(vals) or 1e-9
        means.append(mean); stds.append(sd)
    # Convert raw coefficients to standardized-feature coefficients.
    std_coefs=[float(model.coef_[0][j])*stds[j] for j in range(len(FEATURES))]
    intercept=float(model.intercept_[0] + sum(float(model.coef_[0][j])*means[j] for j in range(len(FEATURES))))
    def score(s):
        z=intercept+sum(std_coefs[j]*((float(s["features"].get(FEATURES[j],0))-means[j])/stds[j]) for j in range(len(FEATURES)))
        return 1/(1+math.exp(max(-35,min(35,-z))))
    rows=sorted([(score(s),s["y"]) for s in oos],reverse=True)
    baseline=(sum(s["y"] for s in oos)/len(oos)) if oos else None
    metrics={"train_n":len(train),"oos_n":len(oos),"oos_baseline":baseline}
    for k in (1,2,3,4,5,6,10):
        sel=rows[:k]; metrics[f"top{k}"]={"selected":len(sel),"hits":sum(y for _,y in sel),"precision":(sum(y for _,y in sel)/len(sel) if sel else None)}
    # Overall OOS ranking precision for all events, not a threshold classifier.
    metrics["oos_auc"] = None
    try:
        from sklearn.metrics import roc_auc_score
        metrics["oos_auc"]=float(roc_auc_score([s["y"] for s in oos],[score(s) for s in oos])) if len({s["y"] for s in oos})>1 else None
    except Exception: pass
    model_json={"version":f"multi-tier-v2-{level}to{level+1}","level":level,
      "trained_through":TRAIN_END.isoformat(),"features":[],"intercept":intercept,"train_n":len(train)}
    for j,name in enumerate(FEATURES):
        model_json["features"].append({"name":name,"coef":std_coefs[j],"mean":means[j],"std":stds[j]})
    return model_json,metrics

def main():
    universe={str(x["code"]):str(x["name"]) for x in fetch_all_stocks() if is_main_board(str(x.get("code","")),str(x.get("name",""))) and x.get("code")}
    print(f"Current main-board universe: {len(universe)}")
    stock_data={}; failed=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fs={ex.submit(fetch_kline,c,KLINE_COUNT):c for c in universe}
        for fut in as_completed(fs):
            c=futures_code=fs[fut]
            try:
                bars=sorted(fut.result(),key=lambda x:str(x.get("date","")))
                if len(bars)>=MIN_HISTORY+22:
                    idx={str(b.get("date")):i for i,b in enumerate(bars)}; stock_data[c]=(universe[c],idx,bars)
                else: failed+=1
            except Exception: failed+=1
    dates=sorted({datetime.date.fromisoformat(str(b["date"])) for _,_,bars in stock_data.values() for b in bars if b.get("date")})
    dates=[d for d in dates if d<=datetime.date.today()]
    market=market_states(stock_data,dates)
    samples_by_level=defaultdict(list); sample_rows=[]
    # Need exact consecutive trading dates for labels.
    date_pos={d:i for i,d in enumerate(dates)}
    for code,(name,idx,bars) in stock_data.items():
        for d in dates:
            i=idx.get(d.isoformat())
            if i is None or i<MIN_HISTORY or i+1>=len(bars): continue
            if i>=len(bars)-1: continue
            next_date=dates[date_pos[d]+1] if date_pos[d]+1<len(dates) else None
            if next_date is None: continue
            j=idx.get(next_date.isoformat())
            if j!=i+1: continue
            lv=board_level_for_day(bars,i)
            if lv not in LEVELS: continue
            feat=make_features(bars,i,market,d,lv)
            if feat is None: continue
            y=int(board_level_for_day(bars,j)>=lv+1)
            s={"date":d.isoformat(),"prediction_date":next_date.isoformat(),"code":code,"name":name,"level":lv,"y":y,"features":feat}
            samples_by_level[lv].append(s); sample_rows.append(s)
    results={"train_end":TRAIN_END.isoformat(),"oos_start":OOS_START.isoformat(),
             "data_source":"Sina daily K-line","universe":"沪深主板 only; exclude ST/STAR/ChiNext/Beijing",
             "failed_kline_fetches":failed,"levels":{}}
    models={}
    for lv in LEVELS:
        samples=samples_by_level[lv]
        fitted=fit_one_level(samples,lv)
        base=sum(s["y"] for s in samples)/len(samples) if samples else None
        results["levels"][str(lv)]={"sample_n":len(samples),"baseline":base,
          "train_n":sum(s["date"]<=TRAIN_END.isoformat() for s in samples),
          "oos_n":sum(s["date"]>=OOS_START.isoformat() for s in samples)}
        if fitted:
            models[str(lv)]=fitted[0]; results["levels"][str(lv)].update(fitted[1])
        else:
            results["levels"][str(lv)]["model_status"]="insufficient_samples"
    MODEL_OUT.parent.mkdir(parents=True,exist_ok=True); MODEL_OUT.write_text(json.dumps(models,ensure_ascii=False,indent=2),encoding="utf-8")
    OUT_JSON.parent.mkdir(parents=True,exist_ok=True); OUT_JSON.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    with OUT_CSV.open("w",encoding="utf-8-sig",newline="") as fh:
        w=csv.writer(fh); w.writerow(["date","prediction_date","level","code","name","label_next_level"])
        for s in sample_rows: w.writerow([s["date"],s["prediction_date"],s["level"],s["code"],s["name"],s["y"]])
    print(json.dumps(results,ensure_ascii=False,indent=2))
if __name__=="__main__": raise SystemExit(main())
