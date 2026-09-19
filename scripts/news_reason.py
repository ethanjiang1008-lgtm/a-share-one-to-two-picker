#!/usr/bin/env python3
"""当日涨停原因证据层。

职责：只负责收集和归纳事件证据，不进入 V1 概率模型。
数据源：巨潮资讯公告、东方财富个股新闻、财联社快讯。
"""
from __future__ import annotations

import html
import json
import re
import ssl
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
HEADERS = {"User-Agent": UA, "Referer": "https://www.cninfo.com.cn/"}
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

CATALOG = {
    "业绩": ["业绩预增", "业绩预告", "业绩快报", "净利润", "营收", "增长", "扭亏", "盈利"],
    "订单": ["订单", "中标", "中签", "合同", "采购", "供货", "交付", "客户"],
    "合作": ["合作", "战略合作", "签约", "协议", "联合", "生态"],
    "产品技术": ["发布", "新品", "产品", "技术", "突破", "量产", "研发", "专利"],
    "政策": ["政策", "规划", "意见", "指导意见", "行动方案", "通知", "实施方案", "国务院", "部委"],
    "并购重组": ["并购", "收购", "重组", "资产注入", "定增", "发行股份"],
    "股东资本": ["回购", "增持", "减持", "股东大会", "股权激励"],
    "价格行业": ["涨价", "提价", "价格上涨", "供给", "停产", "库存", "产能"],
    "事件": ["事故", "地震", "停电", "制裁", "关税", "出口", "召回"],
}
SUSTAINABILITY = {
    "高": ["政策", "规划", "行动方案", "订单", "中标", "长期合同", "量产", "战略合作", "行业"],
    "中": ["业绩", "新品", "技术", "回购", "增持", "价格行业", "产品"],
    "低": ["传闻", "市场消息", "媒体解读", "概念", "预计"],
}

def _get(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 12) -> str:
    h = dict(HEADERS)
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST" if data else "GET")
    last = None
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
                raw = r.read()
                for enc in ("utf-8", "gb18030", "gbk"):
                    try:
                        return raw.decode(enc)
                    except UnicodeDecodeError:
                        continue
                return raw.decode("utf-8", errors="ignore")
        except Exception as e:
            last = e
    raise RuntimeError(str(last))

def _clean(s: object) -> str:
    text = html.unescape(str(s or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _date_only(s: object) -> str:
    m = re.search(r"(20\d{2}-\d{1,2}-\d{1,2})", str(s or ""))
    if m:
        return m.group(1)
    m = re.search(r"(20\d{2}/\d{1,2}/\d{1,2})", str(s or ""))
    if m:
        return m.group(1).replace("/", "-")
    return ""

def _same_or_previous_day(value: str, trade_date: str, max_days: int = 1) -> bool:
    d = _date_only(value)
    if not d:
        return False
    try:
        a = datetime.strptime(d, "%Y-%m-%d").date()
        b = datetime.strptime(trade_date, "%Y-%m-%d").date()
        return timedelta(days=0) <= b - a <= timedelta(days=max_days)
    except ValueError:
        return False

def _categories(text: str) -> list[str]:
    found=[]
    for cat, keywords in CATALOG.items():
        if any(k in text for k in keywords):
            found.append(cat)
    return found

def _cninfo(code: str, trade_date: str, size: int = 20) -> list[dict]:
    if code.startswith("6"):
        org_id = f"gssh0{code}"
    else:
        org_id = f"gssz0{code}"
    payload = {
        "stock": f"{code},{org_id}",
        "tabName": "fulltext",
        "pageSize": str(size),
        "pageNum": "1",
        "column": "",
        "category": "",
        "plate": "",
        "seDate": f"{trade_date}~{trade_date}",
        "searchkey": "",
        "secid": "",
        "sortName": "",
        "sortType": "",
        "isHLtitle": "true",
    }
    raw = _get(
        "https://www.cninfo.com.cn/new/hisAnnouncement/query",
        data=urllib.parse.urlencode(payload).encode("utf-8"),
        headers={"Referer": "https://www.cninfo.com.cn/"},
    )
    data=json.loads(raw)
    rows=[]
    for x in (data.get("announcements") or []):
        title=_clean(x.get("announcementTitle"))
        d=_date_only(x.get("announcementTime"))
        if not title:
            continue
        rows.append({
            "type":"公告","title":title,"summary":title,"date":d or trade_date,
            "source":"巨潮资讯","url":x.get("adjunctUrl") or x.get("announcementUrl") or "",
            "categories":_categories(title),"verified_date":d == trade_date
        })
    return rows

def _eastmoney_news(code: str, size: int = 20) -> list[dict]:
    cb="jQuery_news"
    inner=json.dumps({
        "uid":"","keyword":code,"type":["cmsArticleWebOld"],"client":"web",
        "clientType":"web","clientVersion":"curr",
        "param":{"cmsArticleWebOld":{"searchScope":"default","sort":"default",
        "pageIndex":1,"pageSize":size,"preTag":"","postTag":""}},
    },separators=(",",":"))
    params=urllib.parse.urlencode({"cb":cb,"param":inner})
    raw=_get("https://search-api-web.eastmoney.com/search/jsonp?"+params,
             headers={"Referer":"https://so.eastmoney.com/"})
    start=raw.find("("); end=raw.rfind(")")
    if start<0 or end<=start:
        return []
    data=json.loads(raw[start+1:end])
    rows=[]
    articles=((data.get("result") or {}).get("cmsArticleWebOld") or {}).get("list",[]) or []
    for x in articles:
        title=_clean(x.get("title"))
        summary=_clean(x.get("content"))[:320]
        if not title:
            continue
        d=_date_only(x.get("date"))
        rows.append({
            "type":"新闻","title":title,"summary":summary,"date":d,
            "source":_clean(x.get("mediaName")) or "东方财富","url":x.get("url") or "",
            "categories":_categories(title+" "+summary),"verified_date":False
        })
    return rows

def _cls_global(size: int = 120) -> list[dict]:
    raw=_get("https://www.cls.cn/nodeapi/telegraphList?"+urllib.parse.urlencode({"rn":str(size),"page":"1"}),
             headers={"Referer":"https://www.cls.cn/"})
    data=json.loads(raw)
    rows=[]
    for x in ((data.get("data") or {}).get("roll_data") or []):
        title=_clean(x.get("title") or x.get("brief"))
        summary=_clean(x.get("content") or x.get("brief"))
        d=""
        ctime=x.get("ctime")
        if ctime:
            try:
                d=datetime.fromtimestamp(int(ctime)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                d=""
        if title:
            rows.append({"type":"快讯","title":title,"summary":summary[:320],"date":d,
                         "source":"财联社","url":"https://www.cls.cn/","categories":_categories(title+" "+summary),
                         "verified_date":False})
    return rows

def collect_event_evidence(candidates: list[dict], trade_date: str, workers: int = 8) -> dict[str, dict]:
    """对首板候选逐只收集公告/个股新闻，并用全市场快讯补充交叉证据。"""
    global_news=[]
    try:
        global_news=_cls_global(120)
    except Exception:
        global_news=[]

    out={str(x.get("code")): {
        "status":"no_verified_event","confidence":"低","summary":"暂未找到可验证的当日公告/新闻催化",
        "detail":"事件证据层未找到与当日涨停直接匹配的公告或新闻；系统不把概念标签、技术形态或主观猜测当成涨停原因。",
        "verified_evidence":[],"related_evidence":[],"categories":[],"sustainability":"未知","sources":[],
    } for x in candidates}

    def worker(meta):
        code=str(meta.get("code","")); name=str(meta.get("name",""))
        evidence=[]; related=[]
        try:
            for x in _cninfo(code, trade_date):
                evidence.append(x)
        except Exception:
            pass
        try:
            for x in _eastmoney_news(code):
                (evidence if _same_or_previous_day(x.get("date",""), trade_date, 0) else related).append(x)
        except Exception:
            pass

        # 快讯仅在出现代码或公司简称时视为个股相关，避免把行业新闻误配给个股。
        for x in global_news:
            hay=(x.get("title","")+" "+x.get("summary",""))
            if code in hay or (name and len(name)>=2 and name in hay):
                if _same_or_previous_day(x.get("date",""), trade_date, 0):
                    evidence.append(x)
                else:
                    related.append(x)

        # 去重，并给“直接事件”更高优先级。
        uniq={}
        for x in evidence:
            key=(x.get("source"),x.get("title"))
            uniq[key]=x
        evidence=list(uniq.values())
        related_uniq={}
        for x in related:
            key=(x.get("source"),x.get("title"))
            related_uniq[key]=x
        related=list(related_uniq.values())

        alltext=" ".join([x.get("title","")+" "+x.get("summary","") for x in evidence])
        cats=sorted(set(sum([x.get("categories",[]) for x in evidence],[])))
        score=0
        if any(x.get("type")=="公告" for x in evidence): score+=5
        if any(x.get("type")=="新闻" for x in evidence): score+=3
        if any(x.get("type")=="快讯" for x in evidence): score+=2
        if len(evidence)>=2: score+=2
        if len(cats)>=2: score+=1

        if evidence:
            conf="高" if score>=8 else "中" if score>=5 else "低"
            direct_titles="；".join(x["title"] for x in evidence[:4])
            direct_cats="、".join(cats[:5]) if cats else "一般事件"
            summary=f"当日发现 {len(evidence)} 条直接相关证据，核心涉及：{direct_cats}。"
            detail=(f"与 {name}（{code}）当日涨停直接相关的证据包括：{direct_titles}。"
                    f"其中优先级更高的是上市公司公告，其次是个股新闻与快讯；若多个独立来源指向同一事件，可信度提高。")
            sustain="高" if any(cat in alltext for cat in SUSTAINABILITY["高"]) else "中" if any(cat in alltext for cat in SUSTAINABILITY["中"]) else "低"
            return code, {
                "status":"verified","confidence":conf,"summary":summary,"detail":detail,
                "verified_evidence":evidence[:8],"related_evidence":related[:6],"categories":cats,
                "sustainability":sustain,
                "sources":[x.get("url") for x in evidence if x.get("url")]
            }
        if related:
            cats=sorted(set(sum([x.get("categories",[]) for x in related],[])))
            detail=(f"未找到当日直接公告/新闻，但找到 {len(related)} 条前一交易日附近的相关信息。"
                    f"这些信息可作为背景，不足以单独证明今天涨停的直接原因。")
            return code, {
                "status":"background_only","confidence":"低","summary":"只有近期背景信息，没有当日直接催化证据",
                "detail":detail,"verified_evidence":[],"related_evidence":related[:6],"categories":cats,
                "sustainability":"未知","sources":[x.get("url") for x in related if x.get("url")]
            }
        return code,out[code]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(worker,x) for x in candidates]
        for fut in as_completed(fs):
            try:
                code,res=fut.result(); out[code]=res
            except Exception:
                pass
    return out

if __name__ == "__main__":
    print(json.dumps(collect_event_evidence([{"code":"000001","name":"平安银行"}],
                                            datetime.now().strftime("%Y-%m-%d")),ensure_ascii=False,indent=2))
