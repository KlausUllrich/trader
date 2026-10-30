// Trader dashboard: overview (#/) and one page per market (#/m/NAME) with chart and pre-trade check.
"use strict";

const view = document.getElementById("view");
const VOTERS = { trend: "T", breakout: "B", pullback: "P", cot: "C" };
let charts = [];  // [chart, resize observer] pairs, disposed when the page changes

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const side = (d) => (d > 0 ? "Long" : d < 0 ? "Short" : "None");
const dirClass = (d) => (d > 0 ? "up" : d < 0 ? "down" : "muted");

function price(v) {
  if (v == null) return "–";
  const a = Math.abs(v);
  const digits = a >= 1000 ? 0 : a >= 100 ? 2 : a >= 1 ? 3 : 5;
  return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}
const pct = (v, digits = 2) => (v == null ? "–" : `${(v * 100).toFixed(digits)}%`);
const money = (v) => (v == null ? "–" : `$${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`);

function ago(ts) {
  const h = (Date.now() / 1000 - ts) / 3600;
  if (h < 1) return "just now";
  if (h < 48) return `${Math.round(h)}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

function votes(v) {
  return `<span class="votes">${Object.entries(v || {})
    .map(([k, x]) => `<span class="vote v${x}" title="${k}: ${side(x)}">${VOTERS[k] || k[0]}</span>`).join("")}</span>`;
}

function volCell(vol) {
  if (!vol || vol.forecast_pct == null) return `<span class="muted">–</span>`;
  const rank = vol.rank == null ? 0 : vol.rank;
  return `<span class="num">±${vol.forecast_pct.toFixed(2)}%</span> <span class="pill ${vol.regime}">${vol.regime}</span>` +
    `<span class="bar" title="higher than ${Math.round(rank * 100)}% of past forecasts"><i style="width:${rank * 100}%"></i></span>`;
}

function whaleCell(w) {
  if (!w) return `<span class="muted">–</span>`;
  if (w.vote == null) return `<span class="muted">no data</span>`;
  return `<span class="${dirClass(w.vote)}">${w.vote > 0 ? "mostly long" : w.vote < 0 ? "mostly short" : "split"}</span>` +
    ` <span class="muted num">${w.n_long}/${w.n_short}</span>`;
}

function disposeCharts() {
  charts.forEach(([chart, observer]) => { observer.disconnect(); chart.remove(); });
  charts = [];
}

// ---------- overview ----------

async function overview() {
  view.innerHTML = `<p class="muted">Loading markets…</p>`;
  const data = await api("/api/overview");
  document.getElementById("demo").hidden = !data.demo;
  const groups = {};
  data.rows.forEach((r) => (groups[r.group] = groups[r.group] || []).push(r));
  let html = `<p class="note">${esc(data.note)}</p>`;
  for (const [group, rows] of Object.entries(groups)) {
    html += `<h2>${esc(group)}</h2><table class="table"><thead><tr>
      <th>Market</th><th>Price</th><th>1 day</th><th>Expected daily move</th><th>Strategies</th>
      <th>Last signal</th><th>Whales</th><th>Pre-trade check</th></tr></thead><tbody>`;
    for (const r of rows) {
      if (r.error) {
        html += `<tr data-name="${esc(r.name)}"><td class="name">${esc(r.name)}</td><td colspan="7" class="error">${esc(r.error)}</td></tr>`;
        continue;
      }
      const s = r.last_signal;
      const c = r.check;
      html += `<tr data-name="${esc(r.name)}">
        <td class="name">${esc(r.name)}</td>
        <td class="num" data-label="Price">${price(r.price)}</td>
        <td class="num ${r.change_1d >= 0 ? "up" : "down"}" data-label="1 day">${pct(r.change_1d)}</td>
        <td data-label="Expected move">${volCell(r.vol)}</td>
        <td data-label="Strategies">${votes(r.votes)} <span class="${dirClass(r.consensus.direction)}">${r.consensus.direction ? side(r.consensus.direction) : ""}</span></td>
        <td data-label="Last signal">${s ? `<span class="${dirClass(s.direction)}">${side(s.direction)}</span> <span class="muted">${ago(s.time)}</span>` : `<span class="muted">–</span>`}</td>
        <td data-label="Whales">${whaleCell(r.whales)}</td>
        <td data-label="Check">${c ? `<span class="${dirClass(c.direction)}">${side(c.direction)}</span> <span class="pill ${c.verdict}">${c.score}/${c.of} ${c.verdict}</span>` : `<span class="muted">no consensus</span>`}</td>
      </tr>`;
    }
    html += `</tbody></table>`;
  }
  view.innerHTML = html;
  view.querySelectorAll("tbody tr").forEach((tr) => tr.addEventListener("click", () => (location.hash = `#/m/${tr.dataset.name}`)));
}

// ---------- market page ----------

function chartOptions(el) {
  return {
    width: el.clientWidth, height: el.clientHeight,
    layout: { background: { color: css("--panel") }, textColor: css("--muted"), fontFamily: "system-ui, sans-serif" },
    grid: { vertLines: { color: css("--chart-grid") }, horzLines: { color: css("--chart-grid") } },
    rightPriceScale: { borderColor: css("--line") },
    timeScale: { borderColor: css("--line"), timeVisible: true, secondsVisible: false },
    crosshair: { mode: 0 },
  };
}

function makeChart(el) {
  const chart = LightweightCharts.createChart(el, chartOptions(el));
  const observer = new ResizeObserver(() => chart.applyOptions({ width: el.clientWidth, height: el.clientHeight }));
  observer.observe(el);
  charts.push([chart, observer]);
  return chart;
}

async function market(name, tf) {
  disposeCharts();
  view.innerHTML = `<p class="muted">Loading ${esc(name)}…</p>`;
  const d = await api(`/api/chart/${encodeURIComponent(name)}?tf=${tf}`);
  document.getElementById("demo").hidden = !d.demo;
  // Lightweight Charts shows UTC; shift intraday bars to local time.
  const shift = tf === "1h" ? -new Date().getTimezoneOffset() * 60 : 0;
  const t = (x) => x + shift;
  const last = d.candles[d.candles.length - 1];
  const prev = d.candles[d.candles.length - 2];
  const v = d.vol_now;

  view.innerHTML = `
    <div class="market-head">
      <a href="#/" class="muted">← All markets</a>
      <span class="big">${esc(d.name)}</span>
      <span class="num big">${price(last.close)}</span>
      <span class="num ${last.close >= prev.close ? "up" : "down"}">${pct(last.close / prev.close - 1)} last ${tf === "1d" ? "day" : "hour"}</span>
      <span>${votes(d.votes)} <span class="${dirClass(d.consensus.direction)}">${d.consensus.direction ? `${side(d.consensus.direction)} (${d.consensus.agree} agree)` : "no consensus"}</span></span>
    </div>
    <div class="layout">
      <div>
        <div class="panel">
          <div class="toolbar">
            <button data-tf="1d" class="${tf === "1d" ? "on" : ""}">Daily</button>
            <button data-tf="1h" class="${tf === "1h" ? "on" : ""}">1 hour</button>
          </div>
          <div id="main" class="chart"></div>
          <div class="legend">
            <span><i style="background:${css("--accent")}"></i>EMA 50</span>
            <span><i style="background:${css("--warn")}"></i>EMA 200</span>
            <span><i style="background:${css("--up")}"></i>support</span>
            <span><i style="background:${css("--down")}"></i>resistance</span>
            <span>▲▼ consensus signals (number = strategies agreeing)</span>
          </div>
        </div>
        <div class="panel">
          <h3>Expected daily move, next 10 days <span class="muted">(% of price)</span></h3>
          <div id="vol" class="chart small"></div>
          <div class="legend">
            <span><i style="background:${css("--accent")}"></i>forecast at the time</span>
            <span><i style="background:${css("--muted")}"></i>what actually happened</span>
          </div>
        </div>
      </div>
      <div>
        <div class="panel" id="check"></div>
        <div class="panel">
          <h3>Now</h3>
          <dl class="kv">
            <dt>Expected move</dt><dd>${v.forecast_pct == null ? "–" : `±${v.forecast_pct.toFixed(2)}% (${price(v.move)})`} <span class="pill ${v.regime}">${v.regime}</span></dd>
            <dt>Rank vs history</dt><dd>${v.rank == null ? "–" : `higher than ${Math.round(v.rank * 100)}%`}</dd>
            ${d.whales ? `<dt>Whales</dt><dd>${whaleCell(d.whales)}</dd>` : ""}
            ${d.active ? `<dt>Open signal</dt><dd class="${dirClass(d.active.direction)}">${side(d.active.direction)} at ${price(d.active.price)}, ${ago(d.active.time)}</dd>
              <dt>Its stop / target</dt><dd>${price(d.active.stop)} / ${price(d.active.target)}</dd>` : ""}
          </dl>
          ${d.whales && d.whales.vote != null ? `<p class="muted" style="font-size:12px">${esc(d.whales.text)}</p>` : ""}
          <h3 style="margin-top:12px">Support and resistance</h3>
          <dl class="kv">${d.levels.map((l) => `<dt class="${l.kind === "resistance" ? "down" : "up"}">${l.kind}</dt>
            <dd>${price(l.price)} <span class="muted">${l.touches}× · ${l.distance_atr == null ? "" : l.distance_atr.toFixed(1) + " ATR"}</span></dd>`).join("") || "<dt class='muted'>none found</dt><dd></dd>"}</dl>
        </div>
      </div>
    </div>`;
  view.querySelectorAll("[data-tf]").forEach((b) => b.addEventListener("click", () => (location.hash = `#/m/${d.name}/${b.dataset.tf}`)));

  const main = makeChart(document.getElementById("main"));
  const candles = main.addCandlestickSeries({
    upColor: css("--up"), downColor: css("--down"), wickUpColor: css("--up"), wickDownColor: css("--down"), borderVisible: false,
  });
  candles.setData(d.candles.map((c) => ({ ...c, time: t(c.time) })));
  const emaColors = { ema50: css("--accent"), ema200: css("--warn") };
  for (const [k, pts] of Object.entries(d.lines)) {
    const s = main.addLineSeries({ color: emaColors[k], lineWidth: 1, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
    s.setData(pts.map((p) => ({ ...p, time: t(p.time) })));
  }
  candles.setMarkers(d.markers.map((m) => ({
    time: t(m.time), position: m.direction > 0 ? "belowBar" : "aboveBar",
    color: m.direction > 0 ? css("--up") : css("--down"), shape: m.direction > 0 ? "arrowUp" : "arrowDown",
    text: String(m.agree),
  })));
  d.levels.forEach((l) => candles.createPriceLine({
    price: l.price, color: l.kind === "resistance" ? css("--down") : css("--up"), lineWidth: 1, lineStyle: 2,
    axisLabelVisible: false, title: `${l.kind === "resistance" ? "R" : "S"} ${l.touches}×`,
  }));
  if (tf === "1h") main.timeScale().setVisibleLogicalRange({ from: d.candles.length - 300, to: d.candles.length + 5 });
  else main.timeScale().setVisibleLogicalRange({ from: d.candles.length - 180, to: d.candles.length + 3 });

  const volChart = makeChart(document.getElementById("vol"));
  volChart.applyOptions({ timeScale: { visible: false } });
  const actual = volChart.addLineSeries({ color: css("--muted"), lineWidth: 1, priceLineVisible: false, lastValueVisible: false });
  actual.setData(d.vol.filter((p) => p.actual != null).map((p) => ({ time: t(p.time), value: p.actual })));
  const fc = volChart.addLineSeries({ color: css("--accent"), lineWidth: 2, priceLineVisible: false });
  fc.setData(d.vol.map((p) => ({ time: t(p.time), value: p.forecast })));
  // Keep the volatility chart on the same dates as the price chart.
  main.timeScale().subscribeVisibleTimeRangeChange((r) => { if (r) try { volChart.timeScale().setVisibleRange(r); } catch (e) { /* range outside data */ } });

  let planLines = [];
  async function showCheck(direction) {
    const box = document.getElementById("check");
    box.innerHTML = `<h3>Pre-trade check</h3><p class="muted">Checking…</p>`;
    const c = await api(`/api/check/${encodeURIComponent(d.name)}?direction=${direction}`);
    const p = c.plan;
    box.innerHTML = `
      <h3>Pre-trade check</h3>
      <div class="toolbar">
        <button data-dir="long" class="${direction === "long" ? "on" : ""}">Long</button>
        <button data-dir="short" class="${direction === "short" ? "on" : ""}">Short</button>
        <span class="spacer"></span>
        <span class="score">${c.score}/${c.of}</span> <span class="pill ${c.verdict}">${c.verdict}</span>
      </div>
      <dl class="kv">
        <dt>Entry (last close)</dt><dd>${price(p.entry)}</dd>
        <dt>Stop</dt><dd class="down">${price(p.stop)}</dd>
        <dt>Target</dt><dd class="up">${price(p.target)}</dd>
        <dt>Size</dt><dd>${p.units.toLocaleString(undefined, { maximumFractionDigits: p.units < 10 ? 3 : 0 })} units · ${money(p.notional)}</dd>
        <dt>Margin</dt><dd>${money(p.margin)}</dd>
        <dt>Risk if stopped</dt><dd>${money(p.risk)}</dd>
        <dt>Losses to limit</dt><dd>${p.losses_to_limit ?? "–"} in a row</dd>
      </dl>
      <p class="muted" style="font-size:12px;margin:6px 0 0">Stop and target from the ${esc(p.stop_basis)}.</p>
      <ul class="checks">${c.checks.map((x) => `<li><strong>${esc(x.name)}</strong><span class="pill ${x.status}">${x.status}</span><span class="d">${esc(x.detail)}</span></li>`).join("")}</ul>
      <p class="note" style="margin:8px 0 0">${esc(c.note)}</p>`;
    box.querySelectorAll("[data-dir]").forEach((b) => b.addEventListener("click", () => showCheck(b.dataset.dir)));
    planLines.forEach((l) => candles.removePriceLine(l));
    planLines = [
      candles.createPriceLine({ price: p.stop, color: css("--down"), lineWidth: 2, lineStyle: 0, title: "stop" }),
      candles.createPriceLine({ price: p.entry, color: css("--muted"), lineWidth: 1, lineStyle: 0, title: "entry" }),
    ];
    if (p.target != null) planLines.push(candles.createPriceLine({ price: p.target, color: css("--up"), lineWidth: 2, lineStyle: 0, title: "target" }));
  }
  await showCheck(d.consensus.direction < 0 ? "short" : "long");
}

// ---------- routing ----------

async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  try {
    if (parts[0] === "m" && parts[1]) await market(parts[1], parts[2] === "1h" ? "1h" : "1d");
    else { disposeCharts(); await overview(); }
  } catch (e) {
    view.innerHTML = `<p class="error">Could not load: ${esc(e.message)}</p>`;
  }
}

document.getElementById("refresh").addEventListener("click", async (ev) => {
  const b = ev.currentTarget;
  b.disabled = true;
  b.textContent = "Downloading…";
  try { await api("/api/refresh", { method: "POST" }); await route(); }
  catch (e) { alert(`Refresh failed: ${e.message}`); }
  finally { b.disabled = false; b.textContent = "Refresh data"; }
});
window.addEventListener("hashchange", route);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", route);
route();
