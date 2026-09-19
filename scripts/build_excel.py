#!/usr/bin/env python3
from __future__ import annotations
import json,math,shutil,zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
REPORT_DIR=ROOT/"reports"/"daily"
EXCEL_DIR=ROOT/"reports"/"excel"
DOWNLOAD_DIR=ROOT/"docs"/"downloads"
WEB_DATA=ROOT/"docs"/"data"/"latest.json"
SUMMARY=ROOT/"reports"/"evaluation"/"topk_summary.json"

def col(n):
    s=""; n+=1
    while n:
        n,r=divmod(n-1,26); s=chr(65+r)+s
    return s

def cell(ref,v,style=0):
    if v is None or v=="": return f'<c r="{ref}" s="{style}"/>'
    if isinstance(v,bool): return f'<c r="{ref}" s="{style}" t="b"><v>{int(v)}</v></c>'
    if isinstance(v,(int,float)) and math.isfinite(float(v)):
        return f'<c r="{ref}" s="{style}"><v>{v}</v></c>'
    return f'<c r="{ref}" s="{style}" t="inlineStr"><is><t xml:space="preserve">{escape(str(v))}</t></is></c>'

def sheet(rows,widths):
    mx=max((len(r) for r in rows),default=1); body=[]
    for ri,row in enumerate(rows,1):
        cs=[]
        for ci,v in enumerate(row):
            style=0
            if isinstance(v,tuple): style,v=v
            cs.append(cell(f"{col(ci)}{ri}",v,style))
        body.append(f'<row r="{ri}">'+"".join(cs)+"</row>")
    widths_xml="".join(f'<col min="{i+1}" max="{i+1}" width="{w}" customWidth="1"/>' for i,w in enumerate(widths))
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:{col(mx-1)}{len(rows)}"/><cols>{widths_xml}</cols><sheetViews><sheetView workbookViewId="0"><pane ySplit="2" topLeftCell="A3" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><sheetData>{"".join(body)}</sheetData></worksheet>'''

def workbook():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="预测结果" sheetId="1" r:id="rId1"/><sheet name="复盘" sheetId="2" r:id="rId2"/><sheet name="历史TopK" sheetId="3" r:id="rId3"/><sheet name="证据" sheetId="4" r:id="rId4"/></sheets></workbook>'''

def rels():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/package/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'''

def wb_rels():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/><Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet4.xml"/><Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''

def types():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet4.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>'''

def styles():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><numFmts count="1"><numFmt numFmtId="165" formatCode="0.0%"/></numFmts><fonts count="2"><font><sz val="10"/><name val="Aptos"/></font><font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill><fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border><border><left style="thin"><color rgb="FFD0D7DE"/></left><right style="thin"><color rgb="FFD0D7DE"/></right><top style="thin"><color rgb="FFD0D7DE"/></top><bottom style="thin"><color rgb="FFD0D7DE"/></bottom><diagonal/></border></borders><cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="1"/><xf numFmtId="0" fontId="1" fillId="1" borderId="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf><xf numFmtId="0" fontId="1" fillId="2" borderId="1"/><xf numFmtId="0" fontId="0" fillId="0" borderId="1" applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf><xf numFmtId="165" fontId="0" fillId="0" borderId="1"/></cellXfs></styleSheet>'''

def latest():
    fs=sorted(REPORT_DIR.glob("*_one_to_two_v1.json"))
    if not fs: return None,None
    p=fs[-1]; return p,json.loads(p.read_text(encoding="utf-8"))

def pct(x): return ("PCT",float(x or 0))
def wrap(x): return ("WRAP",x or "")

def main():
    src,data=latest()
    if not src or not data.get("rows"):
        print("No daily report; skip Excel."); return 0
    d=data.get("analysis_date") or data.get("date") or "unknown"
    cut=(data.get("information_cutoff") or "").replace("+08:00","").replace(":","").replace("-","").replace(".","")
    EXCEL_DIR.mkdir(parents=True,exist_ok=True); DOWNLOAD_DIR.mkdir(parents=True,exist_ok=True)
    out=EXCEL_DIR/f"{d}_one_to_two_v1_cutoff_{cut}.xlsx"; rows=data["rows"]; mc=data.get("market_context") or {}

    s1=[["A股首板一进二 V1 · 运行结果","","","","","","","","","","","",""],["分析日期",d,"预测日期",data.get("prediction_date"),"信息截止",data.get("information_cutoff"),"V1参考Top1",pct(data.get("reference_test_top1_precision",0))],["排名","代码","股票","价格","当日涨幅","明日连板概率","事件状态","涨停原因","题材标签","持续性","结构判断","主要风险","证据数"]]
    for r in rows:
        er=r.get("event_reason") or {}; theme=er.get("theme_tags") or ""; theme="、".join(theme) if isinstance(theme,list) else str(theme)
        ev=(er.get("verified_evidence") or [])+(er.get("related_evidence") or [])+(er.get("post_close_evidence") or [])
        s1.append([r.get("rank"),r.get("code"),r.get("name"),r.get("price"),pct((r.get("change_pct") or 0)/100),pct(r.get("score")),er.get("status",""),wrap(er.get("detail") or er.get("summary")),theme,er.get("sustainability","未知"),wrap(r.get("structure_reason")),wrap(r.get("risk")),len(ev)])

    s2=[["复盘：预测后的实际结果","","","","","","","","",""],["排名","代码","股票","V1概率","次日竞价涨幅","次日开盘涨幅","是否二板","实际最高涨幅","实际收盘涨幅","复盘结论"]]
    for r in rows: s2.append([r.get("rank"),r.get("code"),r.get("name"),pct(r.get("score")),None,None,None,None,None,None])

    metrics={}
    if SUMMARY.exists():
        try: metrics=json.loads(SUMMARY.read_text(encoding="utf-8")).get("metrics") or {}
        except Exception: metrics={}
    s3=[["TopK统计：历史基准 + 后续滚动实测","","","","",""],["口径","样本天数","选股数","晋级数","晋级率","状态"]]
    for k in (1,3,6,10):
        m=metrics.get(str(k),{})
        s3.append([f"Top{k}",m.get("days",0),m.get("selected",0),m.get("hits",0),pct(m.get("precision",0)), "尚未积累滚动样本" if not m else "滚动实测"])
    s3.append(["历史参考 Top1","—","—","—",pct(data.get("reference_test_top1_precision",0)),"V1已验证参考值；不是Top6比例"])

    s4=[["排名","代码","股票","证据状态","来源","日期","标题","URL"]]
    for r in rows:
        er=r.get("event_reason") or {}; ev=(er.get("verified_evidence") or [])+(er.get("related_evidence") or [])+(er.get("post_close_evidence") or [])
        for e in ev: s4.append([r.get("rank"),r.get("code"),r.get("name"),er.get("status",""),e.get("source",""),e.get("date",""),e.get("title",""),e.get("url","")])

    sheets=[(s1,[8,12,14,10,10,13,12,42,20,10,42,28,8]),(s2,[8,12,14,13,14,14,12,14,14,22]),(s3,[18,12,12,12,12,24]),(s4,[8,12,14,14,16,12,42,55])]
    with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",types()); z.writestr("_rels/.rels",rels()); z.writestr("xl/workbook.xml",workbook()); z.writestr("xl/_rels/workbook.xml.rels",wb_rels()); z.writestr("xl/styles.xml",styles())
        for i,(rs,ws) in enumerate(sheets,1): z.writestr(f"xl/worksheets/sheet{i}.xml",sheet(rs,ws))
    shutil.copy2(out,DOWNLOAD_DIR/"latest.xlsx")
    data["excel_file"]="./downloads/latest.xlsx"; WEB_DATA.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Excel written: {out}")
    return 0

if __name__=="__main__": raise SystemExit(main())
