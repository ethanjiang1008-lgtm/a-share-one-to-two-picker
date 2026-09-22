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

# V2 uses a shared feature vocabulary, while each board level has its own model.
# New features are deliberately limited to causal, end-of-day information:
# candle anatomy, lagged-day structure, recent limit-up history, and same-tier
# market continuation context.
FEATURES=[
"ret_1d","ret_3d","ret_5d","ret_10d","ret_20d",
"volume_5_vs_20","volume_vs_20d","volume_vs_prev",
"close_above_ma20","ma5_gt_ma10","ma10_gt_ma20","ma20_slope_pct",
"near_high20_pct","near_high60_pct","near_high120_pct","above_low20_pct",
"up_days_5","limit_up_count_5","limit_up_count_10","days_since_prev_limit_up",
"volatility_10d","amplitude_1d","body_ratio","upper_wick_ratio","lower_wick_ratio",
"open_position_in_day","close_position_in_day","close_near_high","intraday_pullback_pct",
"market_first_count","market_2plus_count","market_zt_count",
"market_same_level_count","market_higher_level_count",
"prev_market_first_count","prev_market_2plus_count",
"prev_market_same_level_count","prev_market_higher_level_count",
"prev_market_1to2_rate","prev_market_level_continuation_rate",
"open_gap_pct","open_gap_band_high","open_gap_band_mid","open_gap_band_low",
"open_to_close_pct","amplitude_vs_20d","near_limit_open",
"prior_open_gap_pct","prior_amplitude_pct","prior_volume_vs_20d","prior_close_near_high",
"board_level","board_run_3d","board_run_5d","is_highest_board","is_second_highest_board"
]

BASELINE_CONTINUATION=0.16185320352130533
TUNE_LEVELS={1,2,3}
TUNE_C=[0.1,1.0,10.0]
TUNE_CLASS_WEIGHTS=(None,"balanced")

FEATURE_GROUPS={
    "core":["ret_1d","ret_3d","ret_5d","volume_5_vs_20","volume_vs_20d",
            "close_above_ma20","ma5_gt_ma10","ma10_gt_ma20","amplitude_1d"],
    "momentum":["ret_10d","ret_20d","ma20_slope_pct","near_high20_pct",
                "near_high60_pct","near_high120_pct","above_low20_pct","up_days_5"],
    "limitup":["limit_up_count_5","limit_up_count_10","days_since_prev_limit_up"],
    "candle":["body_ratio","upper_wick_ratio","lower_wick_ratio",
              "open_position_in_day","close_position_in_day","close_near_high",
              "intraday_pullback_pct","prior_amplitude_pct","prior_volume_vs_20d",
              "prior_close_near_high"],
    "market":["market_first_count","market_2plus_count","market_zt_count",
              "market_same_level_count","market_higher_level_count",
              "prev_market_first_count","prev_market_2plus_count",
              "prev_market_same_level_count","prev_market_higher_level_count",
              "prev_market_1to2_rate","prev_market_level_continuation_rate"],
    "regime":["market_heat_ratio","market_first_ratio","market_2plus_ratio",
              "market_zt_change","market_2plus_change","market_first_change",
              "same_level_ratio","higher_level_ratio"],
    "board":["board_level","board_run_3d","board_run_5d",
             "is_highest_board","is_second_highest_board"],
    "opening":["open_gap_pct","open_gap_band_high","open_gap_band_mid",
               "open_gap_band_low","open_to_close_pct","near_limit_open",
               "prior_open_gap_pct"],
    "risk":["volatility_10d","amplitude_vs_20d","volume_vs_prev"]
}

FEATURE_PROFILES={
    1:[("core",),("core","limitup"),("core","regime"),
       ("core","market"),("core","limitup","regime"),
       ("core","market","regime"),("core","limitup","candle","market"),
       ("core","limitup","market","regime")],
    2:[("core",),("core","regime"),("core","market"),
       ("core","limitup","regime"),("core","market","regime"),
       ("core","limitup","candle","market","regime")],
    3:[("core",),("core","candle"),("core","regime"),
       ("core","market"),("core","candle","regime"),
       ("core","limitup","candle","market","regime")],
    4:[("core","candle","market","board")],
    5:[("core","candle","market","board")]
}

def expand_groups(groups):
    out=[]; seen=set()
    for g in groups:
        for name in FEATURE_GROUPS[g]:
            if name not in seen:
                seen.add(name); out.append(name)
    return out

def f(x,d=0.0):
    try:
        v=float(x); return v if math.isfinite(v) else d
    except (TypeError,ValueError):
        return d

def pct(a,b):
    return (a/b-1)*100 if b else 0.0

def limit_up(bar,prev):
    pc=f(prev.get("close"))
    return pc>0 and f(bar.get("close"))/pc-1>=LIMIT_UP

def candle_features(bar, close=0.0):
    o,h,l,cl=f(bar.get("open")),f(bar.get("high")),f(bar.get("low")),f(bar.get("close") if close==0.0 else close)
    rng=max(h-l,1e-9)
    return {
        "amplitude_1d":(h-l)/cl*100 if cl else 0.0,
        "body_ratio":abs(cl-o)/rng,
        "upper_wick_ratio":max(h-max(o,cl),0.0)/rng,
        "lower_wick_ratio":max(min(o,cl)-l,0.0)/rng,
        "open_position_in_day":(o-l)/rng,
        "close_position_in_day":(cl-l)/rng,
        "close_near_high":int(h>0 and (h-cl)/h<=0.002),
        "intraday_pullback_pct":pct(o,l),
    }

def board_level_for_day(bars,i):
    if i==0 or not limit_up(bars[i],bars[i-1]):
        return 0
    level=1
    j=i-1
    while j>0 and limit_up(bars[j],bars[j-1]):
        level+=1
        j-=1
    return level

def market_transition_rate(pv,cur,level):
    prev_levels=pv.get("levels",{})
    cur_levels=cur.get("levels",{})
    cohort={c for c,lv in prev_levels.items() if lv==level}
    if not cohort:
        return BASELINE_CONTINUATION
    hits=sum(1 for c in cohort if cur_levels.get(c,0)>=level+1)
    return hits/len(cohort)

def make_features(bars,i,market,day,board_level,prev_day):
    if i<MIN_HISTORY or i<21 or i==0 or i>=len(bars):
        return None
    b,prev=bars[i],bars[i-1]
    c=[f(x.get("close")) for x in bars[:i+1]]
    v=[f(x.get("volume")) for x in bars[:i+1]]
    if f(b.get("close"))<=0 or f(prev.get("close"))<=0:
        return None

    ma5=statistics.fmean(c[-5:])
    ma10=statistics.fmean(c[-10:])
    ma20=statistics.fmean(c[-20:])
    prev_ma20=statistics.fmean(c[-21:-1])
    v5=statistics.fmean(v[-5:])
    v20=statistics.fmean(v[-20:])

    amps=[
        (f(x.get("high"))-f(x.get("low")))/f(x.get("close"))*100
        for x in bars[max(0,i-19):i+1] if f(x.get("close"))>0
    ]
    avg_amp20=statistics.fmean(amps) if amps else 1.0

    o,h,l,cl=f(b.get("open")),f(b.get("high")),f(b.get("low")),f(b.get("close"))
    pc=f(prev.get("close"))
    current_candle=candle_features(b,cl)

    prev_bar=bars[i-1]
    prev_candle=candle_features(prev_bar)
    prev_prev_close=f(bars[i-2].get("close")) if i>=2 else 0.0
    prev_gap=pct(f(prev_bar.get("open")),prev_prev_close)
    prev_amp=prev_candle["amplitude_1d"]
    prev_vol=f(prev_bar.get("volume"))
    prior_v20=statistics.fmean(v[-21:-1]) if len(v)>=21 else v20
    prior_volume_vs_20d=prev_vol/prior_v20 if prior_v20 else 1.0

    recent_limit_5=sum(
        1 for z in range(max(1,i-4),i)
        if limit_up(bars[z],bars[z-1])
    )
    recent_limit_10=sum(
        1 for z in range(max(1,i-9),i)
        if limit_up(bars[z],bars[z-1])
    )
    days_since_prev_limit_up=99
    for distance,z in enumerate(range(i-1,max(0,i-21),-1),1):
        if limit_up(bars[z],bars[z-1]):
            days_since_prev_limit_up=distance
            break

    cur=market.get(day,{"first":set(),"two":set(),"zt":set(),"levels":{},"max_board":board_level})
    pv=market.get(prev_day,{"first":set(),"two":set(),"zt":set(),"levels":{},"max_board":0})
    prev_first=len(pv.get("first",set()))
    prev_two=len(pv.get("two",set()))
    prev_rate=(len(pv.get("first",set()) & cur.get("two",set()))/prev_first) if prev_first else BASELINE_CONTINUATION

    same_level=sum(1 for lv in cur.get("levels",{}).values() if lv==board_level)
    higher_level=sum(1 for lv in cur.get("levels",{}).values() if lv>board_level)
    prev_same_level=sum(1 for lv in pv.get("levels",{}).values() if lv==board_level)
    prev_higher_level=sum(1 for lv in pv.get("levels",{}).values() if lv>board_level)
    same_level_rate=market_transition_rate(pv,cur,board_level)

    cur_first=len(cur.get("first",set()))
    cur_two=len(cur.get("two",set()))
    cur_zt=len(cur.get("zt",set()))
    prev_zt=len(pv.get("zt",set()))
    heat_ratio=cur_two/cur_zt if cur_zt else 0.0
    first_ratio=cur_first/cur_zt if cur_zt else 0.0
    two_ratio=cur_two/cur_zt if cur_zt else 0.0
    zt_change=cur_zt-prev_zt
    two_change=cur_two-prev_two
    first_change=cur_first-prev_first
    same_ratio=same_level/cur_zt if cur_zt else 0.0
    higher_ratio=higher_level/cur_zt if cur_zt else 0.0

    today_levels=sorted(set(cur.get("levels",{}).values()), reverse=True)
    highest_level=cur.get("max_board",board_level) or board_level
    second_level=today_levels[1] if len(today_levels)>1 else -1

    return {
      "ret_1d":pct(cl,c[-2]),
      "ret_3d":pct(cl,c[-4]),
      "ret_5d":pct(cl,c[-6]),
      "ret_10d":pct(cl,c[-11]),
      "ret_20d":pct(cl,c[-21]),
      "volume_5_vs_20":v5/v20 if v20 else 1.0,
      "volume_vs_20d":f(b.get("volume"))/v20 if v20 else 1.0,
      "volume_vs_prev":f(b.get("volume"))/prev_vol if prev_vol else 1.0,
      "close_above_ma20":int(cl>ma20),
      "ma5_gt_ma10":int(ma5>ma10),
      "ma10_gt_ma20":int(ma10>ma20),
      "ma20_slope_pct":pct(ma20,prev_ma20),
      "near_high20_pct":pct(cl,max(c[-20:])),
      "near_high60_pct":pct(cl,max(c[-60:])),
      "near_high120_pct":pct(cl,max(c[-120:])),
      "above_low20_pct":pct(cl,min(c[-20:])),
      "up_days_5":sum(c[z]>c[z-1] for z in range(len(c)-5,len(c))),
      "limit_up_count_5":recent_limit_5,
      "limit_up_count_10":recent_limit_10,
      "days_since_prev_limit_up":days_since_prev_limit_up,
      "volatility_10d":statistics.pstdev(c[-10:])/cl*100,
      "amplitude_1d":current_candle["amplitude_1d"],
      "body_ratio":current_candle["body_ratio"],
      "upper_wick_ratio":current_candle["upper_wick_ratio"],
      "lower_wick_ratio":current_candle["lower_wick_ratio"],
      "open_position_in_day":current_candle["open_position_in_day"],
      "close_position_in_day":current_candle["close_position_in_day"],
      "close_near_high":current_candle["close_near_high"],
      "intraday_pullback_pct":current_candle["intraday_pullback_pct"],
      "market_first_count":len(cur.get("first",set())),
      "market_2plus_count":len(cur.get("two",set())),
      "market_zt_count":len(cur.get("zt",set())),
      "market_heat_ratio":heat_ratio,
      "market_first_ratio":first_ratio,
      "market_2plus_ratio":two_ratio,
      "market_zt_change":zt_change,
      "market_2plus_change":two_change,
      "market_first_change":first_change,
      "same_level_ratio":same_ratio,
      "higher_level_ratio":higher_ratio,
      "market_same_level_count":same_level,
      "market_higher_level_count":higher_level,
      "prev_market_first_count":prev_first,
      "prev_market_2plus_count":prev_two,
      "prev_market_same_level_count":prev_same_level,
      "prev_market_higher_level_count":prev_higher_level,
      "prev_market_1to2_rate":prev_rate,
      "prev_market_level_continuation_rate":same_level_rate,
      "open_gap_pct":pct(o,pc),
      "open_gap_band_high":int(pct(o,pc)>=8.5),
      "open_gap_band_mid":int(5<=pct(o,pc)<8.5),
      "open_gap_band_low":int(pct(o,pc)<5),
      "open_to_close_pct":pct(cl,o),
      "amplitude_vs_20d":((h-l)/pc*100)/avg_amp20 if pc else 0,
      "near_limit_open":int(pct(o,pc)>=8.0),
      "prior_open_gap_pct":prev_gap,
      "prior_amplitude_pct":prev_amp,
      "prior_volume_vs_20d":prior_volume_vs_20d,
      "prior_close_near_high":prev_candle["close_near_high"],
      "board_level":board_level,
      "board_run_3d":int(board_level>=3),
      "board_run_5d":int(board_level>=5),
      "is_highest_board":int(board_level==highest_level),
      "is_second_highest_board":int(board_level==second_level and board_level<highest_level),
    }

def market_states(stock_data,dates):
    out={}
    for d in dates:
        ds=d.isoformat()
        first=set(); two=set(); zt=set(); levels={}
        for code,(name,idx,bars) in stock_data.items():
            i=idx.get(ds)
            if i is None or i==0 or not limit_up(bars[i],bars[i-1]):
                continue
            zt.add(code)
            lv=board_level_for_day(bars,i)
            levels[code]=lv
            if lv==1:
                first.add(code)
            else:
                two.add(code)
        out[d]={
            "first":first,"two":two,"zt":zt,"levels":levels,
            "max_board":max(levels.values(),default=0)
        }
    return out

def _standardize_fit(samples,feature_names):
    X=[[float(s["features"].get(name,0)) for name in feature_names] for s in samples]
    means=[]; stds=[]
    for j in range(len(feature_names)):
        vals=[row[j] for row in X]
        means.append(statistics.fmean(vals))
        stds.append(statistics.pstdev(vals) or 1e-9)
    Xs=[[(row[j]-means[j])/stds[j] for j in range(len(feature_names))] for row in X]
    return X,Xs,means,stds

def _fit_raw(samples,C,class_weight,feature_names):
    _,Xs,means,stds=_standardize_fit(samples,feature_names)
    y=[s["y"] for s in samples]
    model=LogisticRegression(
        max_iter=3000,
        class_weight=class_weight,
        solver="liblinear",
        C=C
    )
    model.fit(Xs,y)
    return model,means,stds

def _score_model(model,means,stds,s,feature_names):
    z=float(model.intercept_[0])
    vals=s["features"]
    for j,name in enumerate(feature_names):
        z+=float(model.coef_[0][j])*(
            (float(vals.get(name,0))-means[j])/stds[j]
        )
    return 1/(1+math.exp(max(-35,min(35,-z))))

def _score_serialized(model_json,s):
    z=float(model_json["intercept"])
    for item in model_json["features"]:
        z+=float(item["coef"])*(
            (f(s["features"].get(item["name"]))-f(item["mean"]))/
            (f(item["std"]) or 1e-9)
        )
    return 1/(1+math.exp(max(-35,min(35,-z))))

def _daily_metrics(rows):
    by_day=defaultdict(list)
    for row in rows:
        by_day[row["date"]].append(row)
    for d in by_day:
        by_day[d].sort(key=lambda x:(-x["score"],x["code"]))
    result={}
    for k in (1,2,3,4,5,6,10):
        selected=sum(min(k,len(rows)) for rows in by_day.values())
        hits=sum(sum(r["y"] for r in rows[:k]) for rows in by_day.values())
        hit_days=sum(any(r["y"] for r in rows[:k]) for rows in by_day.values())
        result[f"top{k}"]={
            "selected":selected,
            "hits":hits,
            "precision":hits/selected if selected else None,
            "day_hit_days":hit_days,
            "day_hit_rate":hit_days/len(by_day) if by_day else None
        }
    return result

def _segment_metrics(rows, key_name, key_func, min_n=10):
    groups=defaultdict(list)
    for row in rows:
        key=key_func(row)
        if key is not None:
            groups[str(key)].append(row)
    out={}
    for key,items in sorted(groups.items()):
        if len(items)<min_n:
            continue
        m=_daily_metrics(items)
        out[key]={
            "n":len(items),
            "days":len({x["date"] for x in items}),
            "top1":m["top1"]["precision"],
            "top2":m["top2"]["precision"],
            "auc":_auc(items),
        }
    return out

def _add_oos_diagnostics(metrics, scored, train_samples):
    # Diagnostics only; they do not affect model fitting or hyperparameter selection.
    monthly=_segment_metrics(
        scored,"month",
        lambda r:r["date"][:7],
        min_n=20
    )

    # Use training-set medians for fixed, non-leaking regime bands.
    train_zt=sorted(float(s["features"].get("market_zt_count",0)) for s in train_samples)
    train_two=sorted(float(s["features"].get("market_2plus_count",0)) for s in train_samples)
    train_relay=sorted(float(s["features"].get("prev_market_level_continuation_rate",BASELINE_CONTINUATION)) for s in train_samples)

    def q(arr, frac):
        if not arr:
            return 0.0
        pos=(len(arr)-1)*frac
        lo=int(pos); hi=min(lo+1,len(arr)-1); w=pos-lo
        return arr[lo]*(1-w)+arr[hi]*w

    zt_q1,zt_q2=q(train_zt,1/3),q(train_zt,2/3)
    two_q1,two_q2=q(train_two,1/3),q(train_two,2/3)
    relay_q1,relay_q2=q(train_relay,1/3),q(train_relay,2/3)

    def band(v,q1,q2):
        if v<q1: return "low"
        if v<q2: return "mid"
        return "high"

    diagnostics={
        "monthly":monthly,
        "market_zt_band":_segment_metrics(
            scored,"market_zt_band",
            lambda r:band(float(r.get("market_zt_count",0)) if "market_zt_count" in r else 0.0,zt_q1,zt_q2),
            min_n=20
        ),
        "market_2plus_band":_segment_metrics(
            scored,"market_2plus_band",
            lambda r:band(float(r.get("market_2plus_count",0)) if "market_2plus_count" in r else 0.0,two_q1,two_q2),
            min_n=20
        ),
        "prev_level_relay_band":_segment_metrics(
            scored,"prev_level_relay_band",
            lambda r:band(float(r.get("prev_market_level_continuation_rate",BASELINE_CONTINUATION)) if "prev_market_level_continuation_rate" in r else BASELINE_CONTINUATION,relay_q1,relay_q2),
            min_n=20
        ),
        "train_band_thresholds":{
            "market_zt_count":[zt_q1,zt_q2],
            "market_2plus_count":[two_q1,two_q2],
            "prev_market_level_continuation_rate":[relay_q1,relay_q2]
        }
    }
    metrics["oos_diagnostics"]=diagnostics

def _auc(rows):
    try:
        from sklearn.metrics import roc_auc_score
        ys=[r["y"] for r in rows]
        scores=[r["score"] for r in rows]
        return float(roc_auc_score(ys,scores)) if len(set(ys))>1 else None
    except Exception:
        return None

def _split_by_date(samples,train_ratio):
    dates=sorted({s["date"] for s in samples})
    if len(dates)<10:
        return [],[]
    cut=max(1,min(len(dates)-1,int(len(dates)*train_ratio)))
    train_days=set(dates[:cut])
    return [s for s in samples if s["date"] in train_days],[s for s in samples if s["date"] not in train_days]

def choose_hyperparams(train_samples,level):
    default_profile=FEATURE_PROFILES.get(level,[("core",)])[0]
    default={
        "C":1.0,"class_weight":None,"tuned":False,
        "feature_groups":list(default_profile),
        "features":expand_groups(default_profile)
    }
    if level not in TUNE_LEVELS or len(train_samples)<250:
        return default

    dates=sorted({s["date"] for s in train_samples})
    if len(dates)<30:
        return default

    folds=[]
    for train_ratio in (0.60,0.72,0.84):
        tr,va=_split_by_date(train_samples,train_ratio)
        if len(tr)>=100 and len(va)>=20 and len({s["y"] for s in tr})==2:
            folds.append((tr,va))
    if not folds:
        return default

    candidates=[]
    for profile in FEATURE_PROFILES.get(level,[("core",)]):
        feature_names=expand_groups(profile)
        for C in TUNE_C:
            for class_weight in TUNE_CLASS_WEIGHTS:
                fold_scores=[]
                for tr,va in folds:
                    try:
                        model,means,stds=_fit_raw(tr,C,class_weight,feature_names)
                    except Exception:
                        continue
                    scored=[{
                        "date":s["date"],"code":s["code"],"y":s["y"],
                        "score":_score_model(model,means,stds,s,feature_names)
                    } for s in va]
                    m=_daily_metrics(scored)
                    auc=_auc(scored)
                    fold_scores.append((
                        m["top1"]["precision"] or 0.0,
                        m["top2"]["precision"] or 0.0,
                        auc if auc is not None else 0.5
                    ))
                if len(fold_scores)!=len(folds):
                    continue
                avg1=sum(x[0] for x in fold_scores)/len(fold_scores)
                avg2=sum(x[1] for x in fold_scores)/len(fold_scores)
                avga=sum(x[2] for x in fold_scores)/len(fold_scores)
                key=(round(avg1,8),round(avg2,8),round(avga,8))
                candidates.append((key,profile,C,class_weight))
    if not candidates:
        return default
    candidates.sort(key=lambda x:x[0],reverse=True)
    key,profile,C,class_weight=candidates[0]
    return {
        "C":C,"class_weight":class_weight,"tuned":True,
        "feature_groups":list(profile),
        "features":expand_groups(profile),
        "validation_top1":key[0],
        "validation_top2":key[1],
        "validation_auc":key[2]
    }

def _fit_serialized_model(train_samples, level):
    params=choose_hyperparams(train_samples,level)
    feature_names=params["features"]
    model,means,stds=_fit_raw(
        train_samples,float(params["C"]),params["class_weight"],feature_names
    )
    model_json={
        "version":f"multi-tier-v2-{level}to{level+1}",
        "level":level,
        "trained_through":max(s["date"] for s in train_samples),
        "features":[],
        "feature_groups":params["feature_groups"],
        "intercept":float(model.intercept_[0]),
        "train_n":len(train_samples),
        "C":float(params["C"]),
        "class_weight":params["class_weight"],
        "hyperparameter_tuned":bool(params.get("tuned",False))
    }
    for j,name in enumerate(feature_names):
        model_json["features"].append({
            "name":name,
            "coef":float(model.coef_[0][j]),
            "mean":means[j],
            "std":stds[j]
        })
    return model_json,params

def _rolling_oos_score(samples,level):
    oos=[s for s in samples if s["date"]>=OOS_START.isoformat()]
    if not oos:
        return [],[]

    # Monthly expanding-window retraining. Each month's OOS scores only use
    # samples available before that month starts, preventing look-ahead bias.
    months=sorted({s["date"][:7] for s in oos})
    scored=[]
    model_history=[]

    for month in months:
        month_start=f"{month}-01"
        month_oos=[s for s in oos if s["date"][:7]==month]
        if month == OOS_START.isoformat()[:7]:
            train=[s for s in samples if s["date"]<=TRAIN_END.isoformat()]
        else:
            train=[s for s in samples if s["date"]<month_start]
        if len(train)<50 or len({s["y"] for s in train})<2:
            continue
        model_json,params=_fit_serialized_model(train,level)
        # Score using the exact serialized representation that would be deployed.
        for s in month_oos:
            scored.append({
                "date":s["date"],
                "code":s["code"],
                "y":s["y"],
                "score":_score_serialized(model_json,s),
                "market_zt_count":s["features"].get("market_zt_count",0),
                "market_2plus_count":s["features"].get("market_2plus_count",0),
                "prev_market_level_continuation_rate":s["features"].get(
                    "prev_market_level_continuation_rate",BASELINE_CONTINUATION
                )
            })
        model_history.append({
            "month":month,
            "train_n":len(train),
            "feature_groups":params["feature_groups"],
            "features":params["features"],
            "C":params["C"],
            "class_weight":params["class_weight"],
            "validation_top1":params.get("validation_top1"),
            "validation_top2":params.get("validation_top2"),
            "validation_auc":params.get("validation_auc")
        })
    return scored,model_history

def fit_one_level(samples,level):
    initial_train=[s for s in samples if s["date"]<=TRAIN_END.isoformat()]
    oos=[s for s in samples if s["date"]>=OOS_START.isoformat()]
    if len(initial_train)<50 or len({s["y"] for s in initial_train})<2:
        return None

    rolling_enabled=level in TUNE_LEVELS
    if rolling_enabled:
        scored,model_history=_rolling_oos_score(samples,level)
        # Fallback to the original single model if any monthly period could not
        # be trained, preserving robustness for sparse data.
        if len(scored)<len(oos):
            scored=[]
            model_history=[]
            model_json_initial,params_initial=_fit_serialized_model(
                initial_train,level
            )
            for s in oos:
                scored.append({
                    "date":s["date"],
                    "code":s["code"],
                    "y":s["y"],
                    "score":_score_serialized(model_json_initial,s),
                    "market_zt_count":s["features"].get("market_zt_count",0),
                    "market_2plus_count":s["features"].get("market_2plus_count",0),
                    "prev_market_level_continuation_rate":s["features"].get(
                        "prev_market_level_continuation_rate",BASELINE_CONTINUATION
                    )
                })
            model_history=[{
                "month":"initial_fixed_fallback",
                "train_n":len(initial_train),
                "feature_groups":params_initial["feature_groups"],
                "features":params_initial["features"],
                "C":params_initial["C"],
                "class_weight":params_initial["class_weight"],
                "validation_top1":params_initial.get("validation_top1"),
                "validation_top2":params_initial.get("validation_top2"),
                "validation_auc":params_initial.get("validation_auc")
            }]
    else:
        model_json_initial,params_initial=_fit_serialized_model(initial_train,level)
        model_history=[{
            "month":"initial_fixed",
            "train_n":len(initial_train),
            "feature_groups":params_initial["feature_groups"],
            "features":params_initial["features"],
            "C":params_initial["C"],
            "class_weight":params_initial["class_weight"],
            "validation_top1":params_initial.get("validation_top1"),
            "validation_top2":params_initial.get("validation_top2"),
            "validation_auc":params_initial.get("validation_auc")
        }]
        scored=[{
            "date":s["date"],
            "code":s["code"],
            "y":s["y"],
            "score":_score_serialized(model_json_initial,s),
            "market_zt_count":s["features"].get("market_zt_count",0),
            "market_2plus_count":s["features"].get("market_2plus_count",0),
            "prev_market_level_continuation_rate":s["features"].get(
                "prev_market_level_continuation_rate",BASELINE_CONTINUATION
            )
        } for s in oos]

    baseline=(sum(s["y"] for s in oos)/len(oos)) if oos else None
    metrics={
        "train_n":len(initial_train),
        "oos_n":len(oos),
        "oos_scored_n":len(scored),
        "oos_days":len({s["date"] for s in oos}),
        "oos_baseline":baseline,
        "rolling_retrain":rolling_enabled,
        "retrain_history":model_history
    }
    if model_history:
        metrics["selected_hyperparameters"]=model_history[-1]
    metrics.update(_daily_metrics(scored))
    metrics["oos_auc"]=_auc(scored)
    _add_oos_diagnostics(metrics,scored,initial_train)

    # Final deployable model is always retrained on the latest available
    # labeled sample, so the live prediction is not frozen at 2026-07-08.
    final_train=[s for s in samples if s["date"]<=max(s["date"] for s in samples)]
    final_model,final_params=_fit_serialized_model(final_train,level)
    final_model["rolling_retrain"]=rolling_enabled
    final_model["retrain_history"]=model_history
    return final_model,metrics

def build_today_prediction(stock_data, models, dates, market):
    if not dates:
        return None
    analysis_date=max(dates)
    pos=dates.index(analysis_date)
    from trading_calendar import next_trading_day
    pred_date=(
        dates[pos+1] if pos+1<len(dates)
        else next_trading_day(analysis_date)
    ).isoformat()

    today_by_code={}
    raw_levels={}
    for code,(name,idx,bars) in stock_data.items():
        i=idx.get(analysis_date.isoformat())
        if i is None or i==0 or i<MIN_HISTORY or not limit_up(bars[i],bars[i-1]):
            continue
        raw_lv=board_level_for_day(bars,i)
        raw_levels[code]=raw_lv
        today_by_code[code]=(name,bars,i,min(raw_lv,6))

    cur={
        "first":{c for c,lv in raw_levels.items() if lv==1},
        "two":{c for c,lv in raw_levels.items() if lv>=2},
        "zt":set(raw_levels),
        "levels":raw_levels,
        "max_board":max(raw_levels.values(),default=0)
    }
    prev_day=dates[pos-1] if pos>0 else None
    pv=market.get(prev_day,{"first":set(),"two":set(),"zt":set(),"levels":{},"max_board":0}) if prev_day else {"first":set(),"two":set(),"zt":set(),"levels":{},"max_board":0}
    market_for_today=dict(market)
    market_for_today[analysis_date]=cur

    buckets={str(k):[] for k in LEVELS}
    for code,(name,bars,i,lv) in today_by_code.items():
        model=models.get(str(lv))
        if not model:
            continue
        feat=make_features(
            bars,i,market_for_today,analysis_date,lv,prev_day
        )
        if feat is None:
            continue
        p=_score_serialized(model,{"features":feat})
        buckets[str(lv)].append({
            "rank":0,"level":lv,"code":code,"name":name,
            "price":f(bars[i].get("close")),"score":p,
            "prediction_date":pred_date
        })

    for k in buckets:
        buckets[k].sort(key=lambda x:(-x["score"],x["code"]))
        for rank,row in enumerate(buckets[k][:10],1):
            row["rank"]=rank
        buckets[k]=buckets[k][:10]

    return {
        "status":"ok",
        "analysis_date":analysis_date.isoformat(),
        "prediction_date":pred_date,
        "model_version":"multi-tier-v2",
        "universe":"沪深主板",
        "market":{
            "first_count":len(cur["first"]),
            "two_plus_count":len(cur["two"]),
            "zt_count":len(cur["zt"]),
            "max_board":cur["max_board"]
        },
        "levels":buckets
    }

def main():
    universe={
        str(x["code"]):str(x["name"])
        for x in fetch_all_stocks()
        if is_main_board(str(x.get("code","")),str(x.get("name","")))
        and x.get("code")
    }
    print(f"Current main-board universe: {len(universe)}")

    stock_data={}; failed=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fs={ex.submit(fetch_kline,c,KLINE_COUNT):c for c in universe}
        for fut in as_completed(fs):
            c=fs[fut]
            try:
                bars=sorted(fut.result(),key=lambda x:str(x.get("date","")))
                if len(bars)>=MIN_HISTORY+22:
                    idx={str(b.get("date")):i for i,b in enumerate(bars)}
                    stock_data[c]=(universe[c],idx,bars)
                else:
                    failed+=1
            except Exception:
                failed+=1

    dates=sorted({
        datetime.date.fromisoformat(str(b["date"]))
        for _,_,bars in stock_data.values()
        for b in bars if b.get("date")
    })
    dates=[d for d in dates if d<=datetime.date.today()]
    market=market_states(stock_data,dates)

    samples_by_level=defaultdict(list)
    sample_rows=[]
    date_pos={d:i for i,d in enumerate(dates)}

    for code,(name,idx,bars) in stock_data.items():
        for d in dates:
            i=idx.get(d.isoformat())
            if i is None or i<MIN_HISTORY or i+1>=len(bars):
                continue
            next_date=dates[date_pos[d]+1] if date_pos[d]+1<len(dates) else None
            if next_date is None:
                continue
            j=idx.get(next_date.isoformat())
            if j!=i+1:
                continue

            lv=board_level_for_day(bars,i)
            if lv not in LEVELS:
                continue
            prev_day=dates[date_pos[d]-1] if date_pos[d]>0 else None
            feat=make_features(bars,i,market,d,lv,prev_day)
            if feat is None:
                continue
            y=int(board_level_for_day(bars,j)>=lv+1)
            s={
                "date":d.isoformat(),
                "prediction_date":next_date.isoformat(),
                "code":code,"name":name,"level":lv,"y":y,
                "features":feat
            }
            samples_by_level[lv].append(s)
            sample_rows.append(s)

    results={
        "train_end":TRAIN_END.isoformat(),
        "oos_start":OOS_START.isoformat(),
        "data_source":"Sina daily K-line",
        "universe":"沪深主板 only; exclude ST/STAR/ChiNext/Beijing",
        "feature_set":"V2 optimized causal daily-structure + tier-context features",
        "failed_kline_fetches":failed,
        "levels":{}
    }
    models={}

    for lv in LEVELS:
        samples=samples_by_level[lv]
        fitted=fit_one_level(samples,lv)
        base=sum(s["y"] for s in samples)/len(samples) if samples else None
        results["levels"][str(lv)]={
            "sample_n":len(samples),
            "baseline":base,
            "train_n":sum(s["date"]<=TRAIN_END.isoformat() for s in samples),
            "oos_n":sum(s["date"]>=OOS_START.isoformat() for s in samples)
        }
        if fitted:
            models[str(lv)]=fitted[0]
            results["levels"][str(lv)].update(fitted[1])
        else:
            results["levels"][str(lv)]["model_status"]="insufficient_samples"

    MODEL_OUT.parent.mkdir(parents=True,exist_ok=True)
    MODEL_OUT.write_text(
        json.dumps(models,ensure_ascii=False,indent=2),
        encoding="utf-8"
    )

    today_prediction=build_today_prediction(stock_data,models,dates,market)
    latest_path=ROOT/"docs"/"data"/"multi_tier_latest.json"
    if today_prediction:
        latest_path.parent.mkdir(parents=True,exist_ok=True)
        latest_path.write_text(
            json.dumps(today_prediction,ensure_ascii=False,indent=2),
            encoding="utf-8"
        )
        results["today_prediction"]=today_prediction

    OUT_JSON.parent.mkdir(parents=True,exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(results,ensure_ascii=False,indent=2),
        encoding="utf-8"
    )

    with OUT_CSV.open("w",encoding="utf-8-sig",newline="") as fh:
        w=csv.writer(fh)
        w.writerow(["date","prediction_date","level","code","name","label_next_level"])
        for s in sample_rows:
            w.writerow([
                s["date"],s["prediction_date"],s["level"],
                s["code"],s["name"],s["y"]
            ])

    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__":
    raise SystemExit(main())
