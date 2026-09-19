"""新浪财经公开行情数据适配层。"""
from __future__ import annotations
import json, ssl, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36","Referer":"https://finance.sina.com.cn"}
SSL_CTX=ssl.create_default_context(); SSL_CTX.check_hostname=False; SSL_CTX.verify_mode=ssl.CERT_NONE
DEFAULT_TIMEOUT=15; RETRIES=3; BACKOFF_SECONDS=(1.0,2.0,3.0); RANK_PAGE_SIZE=100; MAX_RANK_PAGES=60

def _fetch_url(url:str, timeout:int=DEFAULT_TIMEOUT, label:str="")->str:
    req=urllib.request.Request(url,headers=HEADERS); last=None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(req,timeout=timeout) as resp: return resp.read().decode("utf-8")
        except Exception as exc: last=exc
        try:
            with urllib.request.urlopen(req,timeout=timeout,context=SSL_CTX) as resp: return resp.read().decode("utf-8")
        except Exception as exc: last=exc
        if attempt<RETRIES-1: time.sleep(BACKOFF_SECONDS[attempt])
    raise RuntimeError(f"Sina request failed after {RETRIES} attempts: {label}") from last

def is_main_board(code:str,name:str="")->bool:
    name=(name or "").strip().upper()
    if not code:
        return False
    # 明确排除 ST / *ST / S*ST / SST、退市股。
    if name.startswith(("ST","*ST","S*ST","SST")) or "退" in name:
        return False
    return code.startswith(("600","601","603","605","000","001","002","003"))

def _norm(item):
    def f(k,d=0.0):
        try: return float(item.get(k,d) or d)
        except (TypeError,ValueError): return d
    return {"code":str(item.get("code","")).strip(),"symbol":str(item.get("symbol","")).strip(),"name":str(item.get("name","")).strip(),"price":f("trade"),"change_pct":f("changepercent"),"change_amt":f("pricechange"),"volume":f("volume"),"amount":f("amount"),"open":f("open"),"high":f("high"),"low":f("low"),"prev_close":f("settlement"),"turnover_rate":f("turnoverratio"),"total_mcap":f("mktcap"),"circ_mcap":f("nmc"),"pe":f("per"),"pb":f("pb")}

def _rank_page(sort,page,asc=0,num=RANK_PAGE_SIZE):
    url=("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?"
         f"page={page}&num={num}&sort={sort}&asc={asc}&node=hs_a&symbol=&_s_r_a=sort")
    data=json.loads(_fetch_url(url,15,f"rank:{sort}:{page}"))
    return [_norm(x) for x in data] if isinstance(data,list) else []

def fetch_all_stocks():
    rows=[]; seen=set()
    for page in range(1,MAX_RANK_PAGES+1):
        batch=_rank_page("changepercent",page,0)
        if not batch: break
        for row in batch:
            code=row.get("code","")
            if code and code not in seen: seen.add(code); rows.append(row)
        if len(batch)<RANK_PAGE_SIZE: break
        time.sleep(0.1)
    return rows

def _parse(data):
    if not isinstance(data,list): return []
    out=[]
    for x in data:
        if not isinstance(x,dict): continue
        try: out.append({"date":x.get("day",""),"open":float(x.get("open",0) or 0),"close":float(x.get("close",0) or 0),"high":float(x.get("high",0) or 0),"low":float(x.get("low",0) or 0),"volume":float(x.get("volume",0) or 0),"turnover":float(x.get("turnover",0) or 0),"amount":float(x.get("amount",0) or 0)})
        except (TypeError,ValueError): pass
    return out

def fetch_kline(code:str,count:int=80):
    market="sh" if code.startswith("6") else "sz"
    url=("https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?"
         f"symbol={market}{code}&scale=240&ma=no&datalen={count}")
    return _parse(json.loads(_fetch_url(url,15,f"kline:{code}")))

def fetch_klines_parallel(codes,count=80,workers=10):
    out={}; ok=fail=0
    if not codes: return out,{"requested":0,"success":0,"failed":0}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fs={ex.submit(fetch_kline,c,count):c for c in codes}
        for fut in as_completed(fs):
            c=fs[fut]
            try:
                bars=fut.result(); out[c]=bars; ok+=int(bool(bars)); fail+=int(not bool(bars))
            except Exception: out[c]=[]; fail+=1
    return out,{"requested":len(codes),"success":ok,"failed":fail}

if __name__=="__main__":
    rows=[r for r in fetch_all_stocks() if is_main_board(r.get("code",""),r.get("name",""))]
    print(json.dumps(rows[:20],ensure_ascii=False,indent=2))