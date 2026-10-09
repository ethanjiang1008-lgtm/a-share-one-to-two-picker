const TIER_LABELS = {"1":"一进二","2":"二进三","3":"三进四","4":"四进五","5":"五进六","6":"六进七+"};
const fmtPct = v => Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(2) + "%" : "—";
const fmtNum = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString("zh-CN") : "—";
const fmtAuc = v => Number.isFinite(Number(v)) ? Number(v).toFixed(3) : "—";
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function metricPct(level, key) {
  const v = level?.[key]?.precision;
  return v === null || v === undefined ? "—" : fmtPct(v);
}
function renderStatus(report, prediction) {
  const fields = [["训练截止", report.train_end || "—"],["样本外起始", report.oos_start || "—"],["最近分析日", prediction.analysis_date || "—"],["下一预测日", prediction.prediction_date || "—"]];
  document.getElementById("v2Status").innerHTML = fields.map(([label, value]) =>
    '<div><span class="label">' + esc(label) + '</span><strong>' + esc(value) + '</strong></div>').join("");
}
function renderSummary(report, prediction) {
  const levels = Object.values(report.levels || {});
  const totalSamples = levels.reduce((sum, x) => sum + (Number(x.sample_n) || 0), 0);
  const totalOos = levels.reduce((sum, x) => sum + (Number(x.oos_n) || 0), 0);
  const cards = [["回测总样本",fmtNum(totalSamples)],["样本外样本",fmtNum(totalOos)],["样本外起始",report.oos_start || "—"],["当日涨停数量",fmtNum(prediction.market?.zt_count)],["当日首板数量",fmtNum(prediction.market?.first_count)],["当日最高板",prediction.market?.max_board ? prediction.market.max_board + " 板" : "—"]];
  document.getElementById("v2BacktestSummary").innerHTML = cards.map(([label, value]) =>
    '<article class="stat"><span>' + esc(label) + '</span><strong>' + esc(value) + '</strong></article>').join("");
}
function renderBacktest(report) {
  const rows = [];
  for (const level of ["1","2","3","4","5","6"]) {
    const m = report.levels?.[level];
    if (!m) continue;
    rows.push('<tr><td><strong>' + esc(TIER_LABELS[level]) + '</strong></td>' +
      '<td>' + fmtNum(m.sample_n) + '</td><td>' + fmtNum(m.train_n) + '</td><td>' + fmtNum(m.oos_n) + '</td>' +
      '<td>' + fmtPct(m.oos_baseline) + '</td>' +
      '<td><strong>' + metricPct(m,"top1") + '</strong><div class="muted">' + fmtNum(m.top1?.hits) + ' / ' + fmtNum(m.top1?.selected) + '</div></td>' +
      '<td>' + metricPct(m,"top2") + '<div class="muted">' + fmtNum(m.top2?.hits) + ' / ' + fmtNum(m.top2?.selected) + '</div></td>' +
      '<td>' + metricPct(m,"top3") + '<div class="muted">' + fmtNum(m.top3?.hits) + ' / ' + fmtNum(m.top3?.selected) + '</div></td>' +
      '<td>' + fmtPct(m.top1?.day_hit_rate) + '<div class="muted">' + fmtNum(m.top1?.day_hit_days) + ' / ' + fmtNum(m.oos_days) + ' 天</div></td>' +
      '<td>' + fmtAuc(m.oos_auc) + '</td></tr>');
  }
  document.getElementById("v2BacktestRows").innerHTML = rows.length ? rows.join("") : '<tr><td colspan="10">回测报告中没有可显示的板级结果。</td></tr>';
}
function renderMonthly(report) {
  const rows = [];
  for (const level of ["1","2","3","4","5","6"]) {
    const monthly = report.levels?.[level]?.oos_diagnostics?.monthly || {};
    for (const month of Object.keys(monthly).sort()) {
      const m = monthly[month];
      rows.push('<tr><td>' + esc(TIER_LABELS[level]) + '</td><td>' + esc(month) + '</td><td>' + fmtNum(m.n) + '</td>' +
        '<td>' + fmtNum(m.days) + '</td><td><strong>' + fmtPct(m.top1) + '</strong></td><td>' + fmtPct(m.top2) + '</td><td>' + fmtAuc(m.auc) + '</td></tr>');
    }
  }
  document.getElementById("v2MonthlyRows").innerHTML = rows.length ? rows.join("") : '<tr><td colspan="7">当前回测报告还没有分月统计。</td></tr>';
}
function renderPredictions(prediction) {
  const meta = [["分析日期",prediction.analysis_date || "—"],["预测日期",prediction.prediction_date || "—"],["涨停总数",prediction.market?.zt_count ?? "—"],["最高板",prediction.market?.max_board ? prediction.market.max_board + " 板" : "—"]];
  document.getElementById("v2PredictionMeta").innerHTML = meta.map(([label,value]) =>
    '<div class="market-item"><div class="name">' + esc(label) + '</div><div class="value">' + esc(value) + '</div></div>').join("");
  document.getElementById("v2PredictionCards").innerHTML = ["1","2","3","4","5","6"].map(level => {
    const items = prediction.levels?.[level] || [];
    const body = items.length ? items.map((r,i) =>
      '<div class="v2-row"><div><strong>TOP ' + (i+1) + '</strong> ' + esc(r.name) + ' <span class="code">' + esc(r.code) + '</span>' +
      '<div class="muted">价格：' + (Number.isFinite(Number(r.price)) ? Number(r.price).toFixed(2) : "—") + ' · 排名评分</div></div><strong>' + fmtPct(r.score) + '</strong></div>').join("") :
      '<div class="block-text muted">本次没有可用候选。</div>';
    return '<article class="card"><div class="section-head" style="margin:0 0 12px"><div><div class="eyebrow">LEVEL ' + level + '</div><h3>' + esc(TIER_LABELS[level]) + '</h3></div><div class="hint">' + items.length + ' 个候选</div></div>' + body + '</article>';
  }).join("");
}
async function boot() {
  try {
    const [reportRes,predictionRes] = await Promise.all([
      fetch("./data/multi_tier_v2_backtest.json?t=" + Date.now(),{cache:"no-store"}),
      fetch("./data/multi_tier_latest.json?t=" + Date.now(),{cache:"no-store"})
    ]);
    if (!reportRes.ok) throw new Error("V2 回测结果 HTTP " + reportRes.status);
    if (!predictionRes.ok) throw new Error("V2 最新预测 HTTP " + predictionRes.status);
    const [report,prediction] = await Promise.all([reportRes.json(),predictionRes.json()]);
    renderStatus(report,prediction);
    renderSummary(report,prediction);
    renderBacktest(report);
    renderMonthly(report);
    renderPredictions(prediction);
  } catch (e) {
    document.getElementById("v2Status").innerHTML = '<div class="block-text muted">暂时无法读取 V2 最新结果：' + esc(e.message || e) + '</div>';
    document.getElementById("v2BacktestRows").innerHTML = '<tr><td colspan="10">请确认 V2 工作流已成功运行并完成网页部署。</td></tr>';
    document.getElementById("v2MonthlyRows").innerHTML = '<tr><td colspan="7">尚无可读取的分月数据。</td></tr>';
    document.getElementById("v2PredictionCards").innerHTML = '<article class="card"><div class="block-text muted">暂无最新预测数据。</div></article>';
  }
}
boot();
