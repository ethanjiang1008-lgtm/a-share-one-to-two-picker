#!/usr/bin/env python3
from __future__ import annotations
import csv,json,math,zipfile,shutil
from pathlib import Path
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
REPORT_DIR=ROOT/"reports"/"daily"
EVAL_DIR=ROOT/"reports"/"evaluation"
EXCEL_DIR=ROOT/"reports"/"excel"
DOWNLOAD_DIR=ROOT/"docs"/"downloads"
WEB_DATA=ROOT/"docs"/"data"/"latest.json"
SUMMARY=EVAL_DIR/"topk_summary.json"
RECORDS=EVAL_DIR/"daily_topk_validation.csv"
OUTPUT=EXCEL_DIR/"a_share_one_to_two_review.xlsx"

def col(n):
    s=""; n+=1
    while n:
        n,r=divmod(n-1,26); s=chr(65+r)+s
    return s

def fmt(v):
    if v is None or v=="": return ("",0)
    if isinstance(v,tuple): return v
    return v,0

def cell(ref,v):
    style=0
    if isinstance(v,tuple):
        v,style=v
    if v is None or v=="":
        return f'<c r="{ref}" s="{style}"/>'
    if isinstance(v,bool):
        return f'<c r="{ref}" s="{style}" t="b"><v>{int(v)}</v></c>'
    if isinstance(v,(int,float)):
        if not math.isfinite(float(v)): return f'<c r="{ref}" s="{style}"/>'
        return f'<c r="{ref}" s="{style}"><v>{v}</v></c>'
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>'

def pct(v): return (v,4)
def wrap(v): return (v or "",3)
def header(v): return (v,1)
def title(v): return (v,2)

def sheet(rows,widths,freeze=2):
    mx=max((len(r) for r in rows),default=1)
    body=[]
    for ri,row in enumerate(rows,1):
        cs=[]
        for ci,v in enumerate(row):
            cs.append(cell(f"{col(ci)}{ri}",v))
        body.append(f'<row r="{ri}">{"".join(cs)}</row>')
    widths_xml="".join(f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>' for i,w in enumerate(widths))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:{col(mx-1)}{len(rows)}"/><cols>{widths_xml}</cols><sheetViews><sheetView workbookViewId="0"><pane ySplit="{freeze}" topLeftCell="A{freeze+1}" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetData>{"".join(body)}</sheetData></worksheet>'''

def workbook_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="每日预测" sheetId="1" r:id="rId1"/><sheet name="复盘验证" sheetId="2" r:id="rId2"/><sheet name="历史TopK" sheetId="3" r:id="rId3"/><sheet name="证据" sheetId="4" r:id="rId4"/></sheets></workbook>'''

def rels():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'''

def wb_rels():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/><Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet4.xml"/><Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''

def types():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet4.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>'''

def styles():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="165" formatCode="0.0%"/></numFmts><fonts count="3"><font><sz val="10"/><name val="Aptos"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font><font><b/><sz val="13"/><name val="Aptos"/></font></fonts><fills count="4"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFEAF2F8"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left style="thin"><color rgb="FFD0D7DE"/></left><right style="thin"><color rgb="FFD0D7DE"/></right><top style="thin"><color rgb="FFD0D7DE"/></top><bottom style="thin"><color rgb="FFD0D7DE"/></bottom><diagonal/></border></borders><cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="1"/><xf numFmtId="0" fontId="1" fillId="1" borderId="1"/><xf numFmtId="0" fontId="2" fillId="2" borderId="1"/><xf numFmtId="0" fontId="0" fillId="0" borderId="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf><xf numFmtId="165" fontId="0" fillId="0" borderId="1"/></cellXfs></styleSheet>'''

def f(x, default=0.0):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except (TypeError,ValueError):
        return default

def load_json(p):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return None

def all_daily():
    out=[]
    for p in sorted(REPORT_DIR.glob("*_one_to_two_v1.json")):
        d=load_json(p)
        if d and d.get("rows") and d.get("analysis_date"):
            out.append(d)
    return out

def read_reviews():
    if not RECORDS.exists(): return []
    with RECORDS.open(encoding="utf-8-sig",newline="") as fh:
        return list(csv.DictReader(fh))

def main():
    daily=all_daily()
    if not daily:
        print("No daily reports; skip Excel."); return 0
    latest=daily[-1]
    EXCEL_DIR.mkdir(parents=True,exist_ok=True); DOWNLOAD_DIR.mkdir(parents=True,exist_ok=True)

    reviews=read_reviews()
    review_by_key={(r.get("prediction_date",""),r.get("code",""),r.get("rank","")):r for r in reviews}

    # One long-term workbook: each analysis date appears as a grouped block.
    s1=[
      [title("A股首板一进二 V1 · 长期预测台账")]+[""]*11,
      ["说明","同一分析日重复运行：对应日期的预测记录以最新一次运行为准；不同日期持续累积。"]+[""]*10,
      [header(x) for x in ["分析日期","预测日期","排名","代码","股票","价格","当日涨幅","明日连板概率","事件状态","涨停原因","结构判断","主要风险"]]
    ]
    for d in daily:
        for r in sorted(d["rows"],key=lambda x:int(x.get("rank",999))):
            er=r.get("event_reason") or {}
            s1.append([d.get("analysis_date"),d.get("prediction_date"),r.get("rank"),r.get("code"),r.get("name"),r.get("price"),
                       pct(f(r.get("change_pct",0))/100),pct(f(r.get("score",0))),er.get("status",""),wrap(er.get("detail") or er.get("summary")),
                       wrap(r.get("structure_reason")),wrap(r.get("risk"))])

    s2=[
      [title("一进二 · 次日复盘验证（Top10）")]+[""]*12,
      ["说明","只有到预测日实际交易结果已经发生后，才会填入验证数据。当前字段中的“是否二板”是最终收盘确认口径。"]+[""]*11,
      [header(x) for x in ["预测日期","验证日期","排名","代码","股票","V1概率","次日开盘涨幅","次日最高涨幅","次日收盘涨幅","是否二板","复盘结论","验证状态","次日收盘价"]]
    ]
    for r in sorted(reviews,key=lambda x:(x.get("prediction_date",""),int(x.get("rank",999)))):
        s2.append([r.get("prediction_date"),r.get("verification_date"),r.get("rank"),r.get("code"),r.get("name"),
                   pct(f(r.get("score",0))),pct(f(r.get("next_day_open_return",0))) if r.get("next_day_open_return") not in ("",None) else None,
                   pct(f(r.get("next_day_high_return",0))) if r.get("next_day_high_return") not in ("",None) else None,
                   pct(f(r.get("next_day_close_return",0))) if r.get("next_day_close_return") not in ("",None) else None,
                   "是" if str(r.get("is_2board","0"))=="1" else "否",r.get("review_conclusion",""),r.get("review_status","待验证"),r.get("next_day_close")])

    metrics={}
    if SUMMARY.exists():
        data=load_json(SUMMARY) or {}; metrics=data.get("metrics") or {}
    s3=[[title("TopK · 历史基准与滚动实测")]+[""]*6,
        [header(x) for x in ["口径","已验证天数","选股数","晋级数","股票级晋级率","至少命中1只天数","至少命中1只比例"]]]
    for k in (1,3,6,10):
        m=metrics.get(str(k),{})
        s3.append([f"Top{k}",m.get("days",0),m.get("selected",0),m.get("hits",0),pct(f(m.get("precision",0))) if m.get("precision") is not None else None,m.get("day_hit_days",0),
                    pct(f(m.get("day_hit_rate",0))) if m.get("day_hit_rate") is not None else None])
    ref=latest.get("reference_test_top1_precision")
    s3.append(["历史参考Top1","—","—","—",pct(f(ref)) if ref is not None else None,"—","V1已验证参考值；不是Top6比例"])

    s4=[[header(x) for x in ["分析日期","代码","股票","证据状态","来源","日期","标题","URL"]]]
    for d in daily:
        for r in d.get("rows",[]):
            er=r.get("event_reason") or {}
            ev=(er.get("verified_evidence") or [])+(er.get("related_evidence") or [])+(er.get("post_close_evidence") or [])
            for e in ev:
                s4.append([d.get("analysis_date"),r.get("code"),r.get("name"),er.get("status",""),e.get("source",""),e.get("date",""),e.get("title",""),e.get("url","")])

    sheets=[(s1,[13,13,7,12,14,10,11,14,12,44,42,28]),(s2,[13,13,7,12,14,13,14,14,14,10,30,12,13]),(s3,[18,12,12,12,16,18,18]),(s4,[13,12,14,14,16,12,44,55])]
    with zipfile.ZipFile(OUTPUT,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",types()); z.writestr("_rels/.rels",rels()); z.writestr("xl/workbook.xml",workbook_xml()); z.writestr("xl/_rels/workbook.xml.rels",wb_rels()); z.writestr("xl/styles.xml",styles())
        for i,(rows,widths) in enumerate(sheets,1):
            z.writestr(f"xl/worksheets/sheet{i}.xml",sheet(rows,widths))
    shutil.copy2(OUTPUT,DOWNLOAD_DIR/"latest.xlsx")

    # Add the download path to the latest web payload without changing the V1 scores.
    latest_payload=load_json(WEB_DATA) or latest.copy()
    latest_payload["excel_file"]="./downloads/latest.xlsx"
    WEB_DATA.write_text(json.dumps(latest_payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Excel written: {OUTPUT}")
    return 0

if __name__=="__main__": raise SystemExit(main())
