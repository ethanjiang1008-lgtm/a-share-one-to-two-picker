const DATA_URL = "./data/latest.json";
let state = { rows: [] };

const fmtPct = v => Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(1) + "%" : "—";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const confidenceClass = v => v === "高" ? "conf-high" : v === "中" ? "conf-mid" : "conf-low";

function reasonText(row) {
  if (row.event_reason && row.event_reason.status === "verified") {
    return row.event_reason.detail || row.event_reason.summary || "已找到事件证据";
  }
  if (row.event_reason && row.event_reason.status === "inferred") {
    return row.event_reason.detail || row.event_reason.summary || "基于市场联动推断";
  }
  const structure = row.structure_reason || "暂无行情层结构说明。";
  return "事件证据：本次运行未接入可验证的公告/新闻事件层，系统不会编造具体催化。\n行情层观察：" + structure;
}

function modelBasis(row) {
  const positive = (row.model_explanation?.positive || []).slice(0,4).map(x => x.label + "：" + x.text);
  const negative = (row.model_explanation?.negative || []).slice(0,3).map(x => x.label + "：" + x.text);
  const bits = [];
  if (positive.length) bits.push("正向："+positive.join("；"));
  if (negative.length) bits.push("拖累："+negative.join("；"));
  return bits.join(" ") || "暂无可解释因子数据";
}

function evidenceHtml(row) {
  const ev = (row.event_reason?.verified_evidence || []).slice(0,4);
  const post = (row.event_reason?.post_close_evidence || []).slice(0,4);
  const render = x => {
    const url = x.url ? String(x.url) : "";
    const title = esc(x.title || "未命名证据");
    const date = esc(x.date || "");
    const src = esc(x.source || "");
    const summary = x.summary ? '<div class="evidence-summary">'+esc(x.summary)+'</div>' : '';
    return '<div class="evidence"><span class="evidence-meta">'+src+' · '+date+'</span> ' +
      (url ? '<a href="'+esc(url)+'" target="_blank" rel="noopener">'+title+'</a>' : '<span>'+title+'</span>') +
      summary + '</div>';
  };
  let html = ev.length ? ev.map(render).join("") : '<div class="block-text muted">暂无当日直接事件证据。</div>';
  if (post.length) {
    html += '<div class="block-title post-title">分析日收盘后新增</div>' + post.map(render).join("");
  }
  return html;
}

function renderTop(rows) {
  const root = document.getElementById("topList");
  root.innerHTML = rows.slice(0,6).map(row => {
    const p = Number(row.score || 0) * 100;
    return '<article class="card">' +
      '<div class="card-top"><div><div class="rank">TOP ' + esc(row.rank) + '</div>' +
      '<div class="stock">' + esc(row.name) + '<span class="code">' + esc(row.code) + '</span></div></div>' +
      '<div><div class="prob-label">明日连板概率</div><div class="prob">' + p.toFixed(1) + '%</div></div></div>' +
      '<div class="progress"><div style="width:' + Math.min(100,Math.max(0,p)) + '%"></div></div>' +
      '<div class="block"><div class="block-title">当日涨停原因</div><div class="block-text">' + esc(reasonText(row)) + '</div></div>' +
      '<div class="block"><div class="block-title">市场题材归因</div><div class="block-text">' + esc(row.event_reason?.theme_tags || "暂无独立题材标签") + '</div></div>' +
      '<div class="block"><div class="block-title">原因类型</div><div class="block-text">' + esc((row.event_reason?.categories || []).join("、") || "未识别") + '</div></div>' +
      '<div class="block"><div class="block-title">为什么这么判断</div><div class="block-text">' + esc(modelBasis(row)) + '</div></div>' +
      '<div class="block"><div class="block-title">事件证据</div>' + evidenceHtml(row) + '</div>' +
      '<div class="block"><div class="block-title">原因持续性</div><div class="block-text">' + esc(row.event_reason?.sustainability || "未知") + '</div></div>' +
      '<div class="block"><div class="block-title">信息变化</div><div class="block-text">' + esc(row.event_reason?.information_impact || "未发现新增") + '</div></div>' +
      '<div class="block"><div class="block-title">最大风险</div><div class="block-text">' + esc(row.risk || "暂无") + '</div></div>' +
      '<div class="chips"><span class="chip">首板结构</span><span class="chip">市场环境</span><span class="chip">V1模型</span></div>' +
      '</article>';
  }).join("");
}

function renderTable(rows) {
  const q = document.getElementById("searchBox").value.trim().toLowerCase();
  const filtered = rows.filter(r => (r.name+" "+r.code).toLowerCase().includes(q));
  document.getElementById("tableBody").innerHTML = filtered.map(row => {
    const conf = row.event_reason?.confidence || "低";
    return '<tr>' +
      '<td>' + esc(row.rank) + '</td>' +
      '<td><strong>' + esc(row.name) + '</strong><div class="code">' + esc(row.code) + '</div></td>' +
      '<td>' + Number(row.price || 0).toFixed(2) + '</td>' +
      '<td><strong>' + fmtPct(row.score) + '</strong></td>' +
      '<td class="reason">' + esc((row.event_reason?.summary || "暂无")) + '<div class="muted">'+esc((row.event_reason?.categories || []).join("、"))+'</div></td>' +
      '<td class="' + confidenceClass(conf) + '">' + esc(conf) + '<div class="muted">持续性：'+esc(row.event_reason?.sustainability || "未知")+'</div></td>' +
      '<td class="reason">' + esc(modelBasis(row)) + '</td>' +
      '<td class="reason">' + esc(row.risk || "暂无") + '</td>' +
      '</tr>';
  }).join("");
}

function renderMarket(data) {
  const c = data.market_context || {};
  const items = [
    ["今日首板", c.market_first_count],
    ["今日2板+", c.market_2plus_count],
    ["今日涨停", c.market_zt_count],
    ["前日首板", c.prev_market_first_count],
    ["前日2板+", c.prev_market_2plus_count],
    ["前日首板→二板", Number(c.prev_market_1to2_rate || 0) * 100, "%"]
  ];
  document.getElementById("marketGrid").innerHTML = items.map(([n,v,s]) =>
    '<div class="market-item"><div class="name">'+esc(n)+'</div><div class="value">'+(s==="%" ? Number(v).toFixed(1)+"%" : Number(v||0).toFixed(0))+'</div></div>'
  ).join("");
}

async function boot() {
  try {
    const res = await fetch(DATA_URL + "?t=" + Date.now(), { cache: "no-store" });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    if (!data || !data.date || !Array.isArray(data.rows) || !data.rows.length) {
      document.getElementById("emptyState").classList.remove("hidden");
      return;
    }
    state = data;
    document.getElementById("content").classList.remove("hidden");
    document.getElementById("runDate").textContent = data.analysis_date || data.date || "—";
    document.getElementById("predictionDate").textContent = data.prediction_date || "—";
    document.getElementById("cutoffDate").textContent = data.information_cutoff || "—";
    document.getElementById("ztCount").textContent = data.market_context?.market_zt_count ?? "—";
    document.getElementById("firstCount").textContent = data.market_context?.market_first_count ?? "—";
    document.getElementById("twoPlusCount").textContent = data.market_context?.market_2plus_count ?? "—";
    document.getElementById("failedCount").textContent = data.failed ?? "—";
    document.getElementById("footerModel").textContent = "模型：" + (data.model_version || "one-to-two-v1-daily-proxy");
    renderTop(data.rows);
    renderTable(data.rows);
    renderMarket(data);
    document.getElementById("searchBox").addEventListener("input", () => renderTable(state.rows));
  } catch (e) {
    document.getElementById("emptyState").classList.remove("hidden");
    document.getElementById("emptyState").querySelector("p").textContent = "暂无可读取的最新运行数据。";
  }
}
boot();