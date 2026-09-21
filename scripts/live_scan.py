#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from market_data_sina import fetch_all_stocks,fetch_kline,is_main_board
from news_reason import collect_event_evidence
from trading_calendar import next_trading_day,previous_trading_day,is_trading_day
import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/'config/model_v1.json'
STATE_PATH=ROOT/'data/state/market_state.json'
REPORT_DIR=ROOT/'reports/daily'
WEB_DATA_PATH=ROOT/'docs'/'data'/'latest.json'
LIMIT_UP=0.095; MIN_HISTORY=60; WORKERS=10

FEATURE_LABELS={
  'ret_1d':'当日涨幅','ret_3d':'近3日涨幅','ret_5d':'近5日涨幅','ret_10d':'近10日涨幅','ret_20d':'近20日涨幅',
  'volume_5_vs_20':'近5日成交量/20日均量','close_above_ma20':'收盘是否站上20日均线',
  'ma5_gt_ma10':'5日均线是否强于10日均线','ma10_gt_ma20':'10日均线是否强于20日均线',
  'near_high20_pct':'相对20日高点位置','above_low20_pct':'相对20日低点位置','up_days_5':'近5日上涨天数',
  'volatility_10d':'近10日波动率','amplitude_1d':'当日振幅','market_first_count':'当日首板数量',
  'market_2plus_count':'当日2板及以上数量','market_zt_count':'当日涨停总数',
  'prev_market_first_count':'前日首板数量','prev_market_2plus_count':'前日2板及以上数量',
  'prev_market_1to2_rate':'前日首板晋级率','open_gap_pct':'今日开盘涨幅',
  'open_gap_band_high':'今日高开8.5%以上','open_gap_band_mid':'今日高开5%~8.5%',
  'open_gap_band_low':'今日高开不足5%','open_to_close_pct':'今日开盘至收盘涨幅',
  'lower_wick_ratio':'下影线占比','intraday_pullback_pct':'盘中相对开盘回撤',
  'open_position_in_day':'开盘在当日振幅中的位置','amplitude_vs_20d':'振幅/20日平均振幅',
  'volume_vs_20d':'当日成交量/20日均量','near_limit_open':'今日开盘即接近涨停',
  'close_near_high':'收盘接近日内最高价'
}

def f(x,d=0.0):
    try:
        v=float(x); return v if math.isfinite(v) else d
    except (TypeError,ValueError): return d
def pct(a,b): return (a/b-1.0)*100 if b else 0.0
def limit_up(bar,prev): return f(prev.get('close'))>0 and f(bar.get('close'))/f(prev.get('close'))-1>=LIMIT_UP

def base_features(bs,i):
    pre=bs[:i+1]
    if len(pre)<MIN_HISTORY: return {}
    c=[f(x.get('close')) for x in pre]; v=[f(x.get('volume')) for x in pre]; t=[f(x.get('turnover')) for x in pre]; a=[f(x.get('amount')) for x in pre]
    ma5=statistics.fmean(c[-5:]); ma10=statistics.fmean(c[-10:]); ma20=statistics.fmean(c[-20:])
    t5=statistics.fmean(t[-5:]); t20=statistics.fmean(t[-20:]); v5=statistics.fmean(v[-5:]); v20=statistics.fmean(v[-20:]); a5=statistics.fmean(a[-5:]); a20=statistics.fmean(a[-20:])
    price=c[-1]
    return {
      'ret_1d':pct(price,c[-2]),'ret_3d':pct(price,c[-4]),'ret_5d':pct(price,c[-6]),'ret_10d':pct(price,c[-11]),'ret_20d':pct(price,c[-21]),
      'turnover_1d':t[-1],'turnover_5d_avg':t5,'turnover_20d_avg':t20,'turnover_5_vs_20':t5/t20 if t20 else 1,
      'volume_5_vs_20':v5/v20 if v20 else 1,'amount_5_vs_20':a5/a20 if a20 else 1,
      'close_above_ma20':int(price>ma20),'ma5_gt_ma10':int(ma5>ma10),'ma10_gt_ma20':int(ma10>ma20),
      'near_high20_pct':pct(price,max(c[-20:])),'above_low20_pct':pct(price,min(c[-20:])),
      'up_days_5':sum(c[z]>c[z-1] for z in range(len(c)-5,len(c))),
      'volatility_10d':statistics.pstdev(c[-10:])/price*100 if price else 0,
      'amplitude_1d':(f(bs[i].get('high'))-f(bs[i].get('low')))/price*100 if price else 0
    }

def structure_features(bs,i):
    b,prev=bs[i],bs[i-1]; pc=f(prev.get('close')); o=f(b.get('open')); h=f(b.get('high')); l=f(b.get('low')); c=f(b.get('close'))
    rng=max(h-l,1e-9); pre=bs[:i+1]
    vols=[f(x.get('volume')) for x in pre]; amts=[f(x.get('amount')) for x in pre]; trs=[f(x.get('turnover')) for x in pre]
    v20=statistics.fmean(vols[-20:]); a20=statistics.fmean(amts[-20:]); t20=statistics.fmean(trs[-20:])
    amps=[(f(x.get('high'))-f(x.get('low')))/f(x.get('close'))*100 for x in pre[-20:] if f(x.get('close'))>0]
    avg_amp20=statistics.fmean(amps) if amps else 1
    open_gap=pct(o,pc)
    return {
      'open_gap_pct':open_gap,'open_gap_band_high':int(open_gap>=8.5),'open_gap_band_mid':int(5<=open_gap<8.5),'open_gap_band_low':int(open_gap<5),
      'open_to_close_pct':pct(c,o),'lower_wick_ratio':(o-l)/rng,'intraday_pullback_pct':pct(o,l),'open_position_in_day':(o-l)/rng,
      'amplitude_vs_20d':((h-l)/pc*100)/avg_amp20 if pc else 0,'turnover_vs_20d':f(b.get('turnover'))/t20 if t20 else 1,
      'volume_vs_20d':f(b.get('volume'))/v20 if v20 else 1,'amount_vs_20d':f(b.get('amount'))/a20 if a20 else 1,
      'near_limit_open':int(open_gap>=8.0),'close_near_high':int(h>0 and (h-c)/h<=0.002)
    }

def load_json(path,default):
    if not path.exists(): return default
    try: return json.loads(path.read_text(encoding='utf-8'))
    except Exception: return default

def score(row,model):
    z=model['intercept']
    for item in model['features']:
        x=f(row.get(item['name'])); sd=f(item['std'],1e-9)
        z += item['coef']*((x-item['mean'])/sd)
    return 1/(1+math.exp(max(-35,min(35,-z))))

def model_explanation(row, model):
    positive=[]; negative=[]
    for item in model.get('features',[]):
        name=item.get('name')
        sd=f(item.get('std'),1e-9)
        mean=f(item.get('mean'))
        coef=f(item.get('coef'))
        x=f(row.get(name))
        z=(x-mean)/sd
        contribution=coef*z
        if abs(contribution)<0.025:
            continue
        label=FEATURE_LABELS.get(name,name)
        direction='正向' if contribution>0 else '负向'
        text=(f"{label}当前为 {x:.2f}，训练均值为 {mean:.2f}，"
              f"标准化偏离 {z:+.2f}，对模型评分产生{direction}贡献 {contribution:+.3f}")
        item_data={'name':name,'label':label,'value':x,'train_mean':mean,'z':z,
                   'contribution':contribution,'text':text}
        (positive if contribution>0 else negative).append(item_data)
    positive.sort(key=lambda x:x['contribution'], reverse=True)
    negative.sort(key=lambda x:x['contribution'])
    return {'positive':positive[:5],'negative':negative[:5]}

def structure_reason(row, context, market_data_mode):
    bits=[]
    if f(row.get('close_near_high'))>=1:
        if market_data_mode == 'intraday':
            bits.append('截至运行时当前价格贴近日内最高价，盘中价格维持强势。')
        else:
            bits.append('收盘贴近日内最高价，说明涨停日尾盘价格维持强势。')
    if f(row.get('near_limit_open'))>=1:
        bits.append(f"今日开盘已接近涨停价，开盘强度较高（开盘涨幅 {f(row.get('open_gap_pct')):.1f}%）。")
    vol20=f(row.get('volume_vs_20d'),1)
    if vol20>=1.8:
        bits.append(f"截至运行时成交量约为20日均量的 {vol20:.1f} 倍，资金参与度显著放大。")
    elif vol20>=1.2:
        bits.append(f"截至运行时成交量约为20日均量的 {vol20:.1f} 倍，存在明显放量。")
    else:
        bits.append(f"截至运行时成交量约为20日均量的 {vol20:.1f} 倍，量能未出现极端放大。")
    if f(row.get('ma5_gt_ma10')) and f(row.get('ma10_gt_ma20')):
        bits.append('短中期均线保持多头排列，价格结构与趋势方向一致。')
    elif f(row.get('ma5_gt_ma10')):
        bits.append('5日均线仍高于10日均线，但中期趋势强度一般。')
    else:
        bits.append('短期均线未形成明显多头排列，趋势确认度相对有限。')
    bits.append(f"截至本次运行，市场共有 {int(f(context.get('market_zt_count')))} 家涨停，其中首板 {int(f(context.get('market_first_count')))} 家、2板及以上 {int(f(context.get('market_2plus_count')))} 家。")
    return ''.join(bits)

def risk_text(row):
    risks=[]
    if f(row.get('volume_vs_20d'))>=2.2:
        risks.append('当日成交量明显放大，筹码交换较充分')
    if f(row.get('amplitude_1d'))>=11:
        risks.append('当日振幅较大')
    if f(row.get('ret_5d'))>=35:
        risks.append('近5日涨幅偏高，短线获利盘压力更大')
    if f(row.get('ret_10d'))>=55:
        risks.append('近10日涨幅偏高')
    if f(row.get('market_2plus_count'))<5:
        risks.append('市场高阶连板数量偏少')
    if not risks:
        risks.append('主要风险来自次日板块分歧与个股承接强弱')
    return '；'.join(risks)

def runtime_context():
    """Determine the market-data semantics strictly from the actual runtime moment."""
    now=datetime.datetime.now(ZoneInfo("Asia/Shanghai"))
    today=now.date()
    if is_trading_day(today):
        if now.time() < datetime.time(9,25):
            mode='pre_open'
            analysis_date=previous_trading_day(today).isoformat()
        elif now.time() < datetime.time(15,5):
            mode='intraday'
            analysis_date=today.isoformat()
        else:
            mode='daily_close'
            analysis_date=today.isoformat()
    else:
        mode='non_trading'
        analysis_date=previous_trading_day(today).isoformat()
    return {'now':now,'market_data_mode':mode,'analysis_date':analysis_date}

def synthesize_intraday_bar(quote, analysis_date):
    """Build an as-of-now daily bar from the realtime quote for the current trading day."""
    return {
        'date':analysis_date,
        'open':f(quote.get('open')),
        'close':f(quote.get('price')),
        'high':f(quote.get('high')),
        'low':f(quote.get('low')),
        'volume':f(quote.get('volume')),
        'turnover':f(quote.get('turnover_rate')),
        'amount':f(quote.get('amount')),
    }

def resolve_market_context(state, analysis_date, current_two_plus, default_ctx):
    """Resolve previous-trading-day context with safe bootstrap behavior."""
    analysis_day=datetime.date.fromisoformat(analysis_date)
    expected_prev=previous_trading_day(analysis_day).isoformat()
    state_date=str(state.get('last_date') or '')
    state_market=state.get('market') or {}
    current_two_plus_set={str(x) for x in current_two_plus}

    if state_date == analysis_date:
        return {
            'prev_market_first_count':f(state_market.get('market_first_count'),default_ctx['prev_market_first_count']),
            'prev_market_2plus_count':f(state_market.get('market_2plus_count'),default_ctx['prev_market_2plus_count']),
            'prev_market_1to2_rate':f(state_market.get('prev_market_1to2_rate'),default_ctx['prev_market_1to2_rate']),
            'context_source':'same_date_previous_context'
        }

    if state_date == expected_prev:
        prev_codes={str(x) for x in (state.get('first_board_codes') or [])}
        rate=(sum(1 for code in prev_codes if code in current_two_plus_set)/len(prev_codes)
              if prev_codes else default_ctx['prev_market_1to2_rate'])
        return {
            'prev_market_first_count':f(state_market.get('market_first_count'),default_ctx['prev_market_first_count']),
            'prev_market_2plus_count':f(state_market.get('market_2plus_count'),default_ctx['prev_market_2plus_count']),
            'prev_market_1to2_rate':rate,
            'context_source':'previous_trading_day_state'
        }

    return {
        'prev_market_first_count':default_ctx['prev_market_first_count'],
        'prev_market_2plus_count':default_ctx['prev_market_2plus_count'],
        'prev_market_1to2_rate':default_ctx['prev_market_1to2_rate'],
        'context_source':'training_bootstrap',
        'state_last_date':state_date or None,
        'expected_previous_trading_day':expected_prev
    }


def main():
    model=load_json(MODEL_PATH,{})
    default_ctx={'market_first_count':61.51393899639226,'market_2plus_count':12.789603148573303,'market_zt_count':74.30354214496556,
                 'prev_market_first_count':54.58625778943916,'prev_market_2plus_count':12.092981305346015,'prev_market_1to2_rate':0.16185320352130533}
    state=load_json(STATE_PATH,{'last_date':None,'first_board_codes':[],'market':default_ctx})

    run=runtime_context()
    market_data_mode=run['market_data_mode']
    analysis_date=run['analysis_date']
    prediction_date=next_trading_day(datetime.date.fromisoformat(analysis_date)).isoformat()
    cutoff_iso=run['now'].isoformat()

    quotes=[r for r in fetch_all_stocks() if is_main_board(str(r.get('code','')),str(r.get('name','')))]
    candidates=[r for r in quotes if f(r.get('change_pct'))>=9.5 and f(r.get('price'))>0]

    # 开盘前没有“今天首板”可分析：不把前一交易日结果伪装成今天盘中结果。
    if market_data_mode == 'pre_open':
        payload={
            'status':'pre_open',
            'model_version':model.get('version'),
            'reference_test_top1_precision':model.get('reference_test_top1_precision'),
            'trained_through':model.get('trained_through'),
            'analysis_date':analysis_date,
            'prediction_date':prediction_date,
            'date':analysis_date,
            'information_cutoff':cutoff_iso,
            'market_data_mode':market_data_mode,
            'data_source':'Sina',
            'universe':'沪深主板',
            'first_board_count':0,
            'two_plus_count':0,
            'failed':0,
            'market_context':default_ctx,
            'market_context_source':'pre_open',
            'two_plus_codes':[],
            'rows':[]
        }
        WEB_DATA_PATH.parent.mkdir(parents=True,exist_ok=True)
        WEB_DATA_PATH.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(payload,ensure_ascii=False,indent=2))
        return 0

    print(f'运行模式: {market_data_mode}; 分析日期: {analysis_date}; 主板股票数: {len(quotes)}; 当前涨停候选: {len(candidates)}')

    rows=[]; two_plus=[]; failed=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fs={ex.submit(fetch_kline,str(r['code']),80):r for r in candidates}
        for fut in as_completed(fs):
            meta=fs[fut]
            try:
                raw_bars=sorted(fut.result(),key=lambda x:str(x.get('date','')))
                prior=[b for b in raw_bars if str(b.get('date','')) < analysis_date]

                if market_data_mode == 'intraday':
                    if len(prior)<MIN_HISTORY:
                        continue
                    prev=prior[-1]
                    current=synthesize_intraday_bar(meta,analysis_date)
                    bars=prior+[current]
                else:
                    current=next((b for b in raw_bars if str(b.get('date',''))==analysis_date),None)
                    if current is None:
                        continue
                    prior=[b for b in raw_bars if str(b.get('date','')) < analysis_date]
                    if len(prior)<MIN_HISTORY:
                        continue
                    prev=prior[-1]
                    bars=prior+[current]

                if not limit_up(current,prev):
                    continue
                was_two=len(prior)>=2 and limit_up(prev,prior[-2])
                if was_two:
                    two_plus.append(str(meta['code']))
                    continue

                r=dict(meta)
                r['_bars']=bars
                r['_idx']=len(bars)-1
                r['_price']=f(current.get('close'))
                r['_change_pct']=pct(f(current.get('close')),f(prev.get('close')))
                rows.append(r)
            except Exception:
                failed+=1

    first_codes={str(r['code']) for r in rows}
    current_zt=len(rows)+len(two_plus)
    resolved_prev=resolve_market_context(state,analysis_date,two_plus,default_ctx)
    context={
        'market_first_count':len(rows),
        'market_2plus_count':len(two_plus),
        'market_zt_count':current_zt,
        'prev_market_first_count':f(resolved_prev.get('prev_market_first_count'),default_ctx['prev_market_first_count']),
        'prev_market_2plus_count':f(resolved_prev.get('prev_market_2plus_count'),default_ctx['prev_market_2plus_count']),
        'prev_market_1to2_rate':f(resolved_prev.get('prev_market_1to2_rate'),default_ctx['prev_market_1to2_rate']),
        'context_source':resolved_prev.get('context_source','training_bootstrap')
    }

    # 同一交易日重复运行不冻结分数。每次都用本次运行时刻的数据重新计算与排序。
    if not rows:
        payload={
            'status':'no_first_board',
            'model_version':model.get('version'),
            'reference_test_top1_precision':model.get('reference_test_top1_precision'),
            'trained_through':model.get('trained_through'),
            'analysis_date':analysis_date,
            'prediction_date':prediction_date,
            'date':analysis_date,
            'information_cutoff':cutoff_iso,
            'market_data_mode':market_data_mode,
            'data_source':'Sina',
            'universe':'沪深主板',
            'first_board_count':0,
            'two_plus_count':len(two_plus),
            'failed':failed,
            'market_context':context,
            'market_context_source':context.get('context_source'),
            'two_plus_codes':sorted(two_plus),
            'rows':[]
        }
        WEB_DATA_PATH.parent.mkdir(parents=True,exist_ok=True)
        WEB_DATA_PATH.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(payload,ensure_ascii=False,indent=2))
        return 0

    scored=[]
    for item in rows:
        bs=item['_bars']; i=item['_idx']
        row={
            'date':analysis_date,
            'prediction_date':prediction_date,
            'code':str(item['code']),
            'name':str(item['name']),
            'price':item['_price'],
            'change_pct':item['_change_pct']
        }
        row.update(base_features(bs,i))
        row.update(structure_features(bs,i))
        row.update(context)
        row['score']=score(row,model)
        scored.append(row)

    scored.sort(key=lambda x:x['score'],reverse=True)
    event_map=collect_event_evidence(scored,analysis_date,cutoff_iso,workers=8)

    REPORT_DIR.mkdir(parents=True,exist_ok=True)
    model_fields=[x['name'] for x in model['features']]
    extra_fields=[k for k in context.keys() if k not in model_fields]
    fields=['rank','date','prediction_date','code','name','price','change_pct','score']+extra_fields+model_fields
    out=REPORT_DIR/f'{analysis_date}_one_to_two_v1.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields)
        w.writeheader()
        for rank,row in enumerate(scored,1):
            x={k:row.get(k,0) for k in fields}
            x['rank']=rank
            w.writerow(x)

    web_rows=[]
    for rank,row in enumerate(scored,1):
        row['rank']=rank
        web_rows.append({
          'rank':rank,
          'date':analysis_date,
          'prediction_date':prediction_date,
          'code':row['code'],
          'name':row['name'],
          'price':row['price'],
          'change_pct':row['change_pct'],
          'score':row['score'],
          'event_reason':event_map.get(row['code'],{
            'status':'no_verified_event',
            'confidence':'低',
            'summary':'未找到可验证的当日公告/新闻催化',
            'detail':'本次运行没有找到可验证的直接事件证据。',
            'verified_evidence':[],
            'related_evidence':[],
            'categories':[],
            'sustainability':'未知',
            'sources':[]
          }),
          'structure_reason':structure_reason(row,context,market_data_mode),
          'model_explanation':model_explanation(row,model),
          'risk':risk_text(row),
          'factor_snapshot':{k:row.get(k) for k in [x['name'] for x in model.get('features',[])]}
        })

    payload={
      'status':'ok',
      'model_version':model.get('version'),
      'reference_test_top1_precision':model.get('reference_test_top1_precision'),
      'trained_through':model.get('trained_through'),
      'analysis_date':analysis_date,
      'prediction_date':prediction_date,
      'date':analysis_date,
      'information_cutoff':cutoff_iso,
      'market_data_mode':market_data_mode,
      'data_source':'Sina',
      'universe':'沪深主板',
      'first_board_count':len(scored),
      'two_plus_count':len(two_plus),
      'failed':failed,
      'market_context':context,
      'market_context_source':context.get('context_source'),
      'two_plus_codes':sorted(two_plus),
      'rows':web_rows
    }
    WEB_DATA_PATH.parent.mkdir(parents=True,exist_ok=True)
    WEB_DATA_PATH.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    out_json=REPORT_DIR/f'{analysis_date}_one_to_two_v1.json'
    out_json.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    cutoff_key=cutoff_iso.replace("+08:00","").replace(":","").replace("-","").replace(".","")
    snapshot_json=REPORT_DIR/f'{analysis_date}_one_to_two_v1_cutoff_{cutoff_key}.json'
    snapshot_json.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

    print(json.dumps({
        'version':model.get('version'),
        'analysis_date':analysis_date,
        'prediction_date':prediction_date,
        'market_data_mode':market_data_mode,
        'universe':'沪深主板 only',
        'first_board_count':len(scored),
        'two_plus_count':len(two_plus),
        'failed':failed,
        'top10':[{'rank':i+1,'code':r['code'],'name':r['name'],'score':round(r['score'],6)} for i,r in enumerate(scored[:10])]
    },ensure_ascii=False,indent=2))

    save={
        'schema_version':3,
        'last_date':analysis_date,
        'prediction_date':prediction_date,
        'first_board_codes':sorted(first_codes),
        'market':context,
        'context_source':context.get('context_source'),
        'last_run_mode':market_data_mode,
        'last_information_cutoff':cutoff_iso
    }
    STATE_PATH.parent.mkdir(parents=True,exist_ok=True)
    STATE_PATH.write_text(json.dumps(save,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
