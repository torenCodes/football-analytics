// Data Breakdown page: every chart here is plain SVG built from the site's own
// JSON. Live numbers come from data/betting and data/fantasy (rewritten by each
// scan); backtests come from data/lab (written by the local test scripts).

const NS = 'http://www.w3.org/2000/svg';
const $ = id => document.getElementById(id);

function el(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function html(tag, cls, text, parent) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

const linear = (d0, d1, r0, r1) => v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
const num = (v, digits = 2) => v === null || v === undefined ? '—' : v.toFixed(digits).replace('-', '−');
const pct = (v, digits = 1) => v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`;
const signed = (v, digits = 2) => v === null || v === undefined ? '—' : `${v > 0 ? '+' : ''}${num(v, digits)}`;

function corr(xs, ys) {
  const n = xs.length, mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; syy += (ys[i] - my) ** 2; });
  return { r: sxy / Math.sqrt(sxx * syy), slope: sxy / sxx, intercept: my - sxy / sxx * mx };
}

const SERIES = { model: 'var(--series-model)', vegas: 'var(--series-vegas)', muted: 'var(--series-muted)' };

// ---- Tooltip: values lead, labels follow, every string set as text.

const tip = $('viz-tip');

function showTip(x, y, title, rows) {
  tip.replaceChildren();
  html('div', 'viz-tip-title', title, tip);
  for (const r of rows) {
    const row = html('div', 'viz-tip-row', undefined, tip);
    if (r.color) html('span', 'viz-tip-key', undefined, row).style.background = r.color;
    html('strong', '', r.value, row);
    html('span', '', r.label, row);
  }
  tip.hidden = false;
  const box = tip.getBoundingClientRect();
  const left = Math.min(Math.max(8, x + 14), window.innerWidth - box.width - 8);
  const top = y - box.height - 12 < 8 ? y + 16 : y - box.height - 12;
  tip.style.left = `${left + window.scrollX}px`;
  tip.style.top = `${top + window.scrollY}px`;
}

function hideTip() { tip.hidden = true; }

// ---- The frame every chart shares: a plot that redraws at its width, a legend
// for two or more series, and a table twin so no value depends on hovering.

function mount(id, { legend, table, draw, note }) {
  const fig = $(id);
  const plot = html('div', 'viz-plot', undefined, fig);
  if (legend && legend.length > 1) {
    const box = html('div', 'viz-legend', undefined, fig);
    for (const s of legend) {
      const item = html('span', 'viz-legend-item', undefined, box);
      const key = html('span', `viz-key ${s.dot ? 'dot' : ''} ${s.hollow ? 'hollow' : ''}`, undefined, item);
      key.style.setProperty('--key', s.color);
      html('span', '', s.name, item);
    }
  }
  if (note) html('p', 'viz-note', note, fig);
  if (table) {
    const foot = html('div', 'viz-foot', undefined, fig);
    const btn = html('button', 'viz-toggle', 'Show the numbers', foot);
    btn.type = 'button';
    const wrap = html('div', 'table-scroll viz-table', undefined, fig);
    wrap.hidden = true;
    const t = html('table', 'data-table', undefined, wrap);
    const head = html('tr', '', undefined, html('thead', '', undefined, t));
    table.columns.forEach((c, i) => { const th = html('th', i ? 'num-center' : 'left', c, head); th.scope = 'col'; });
    const body = html('tbody', '', undefined, t);
    for (const row of table.rows) {
      const tr = html('tr', '', undefined, body);
      row.forEach((v, i) => html('td', i ? 'num-center' : 'wrap', v, tr));
    }
    btn.addEventListener('click', () => {
      wrap.hidden = !wrap.hidden;
      btn.textContent = wrap.hidden ? 'Show the numbers' : 'Hide the numbers';
    });
  }
  let last = 0;
  const redraw = () => {
    const w = Math.round(plot.clientWidth);
    if (w && w !== last) { last = w; plot.replaceChildren(); draw(plot, w); }
  };
  new ResizeObserver(redraw).observe(plot);
  redraw();
  return () => { last = 0; redraw(); };
}

// ---- Line chart: crosshair snaps to the nearest x, arrow keys walk it.

function lineChart(host, width, o) {
  const m = { t: 18, r: o.endLabels ? 72 : 14, b: 36, l: 44 };
  const H = o.height || 260;
  const svg = el('svg', { width, height: H, role: 'img', 'aria-label': o.aria, tabindex: 0, class: 'viz-svg' }, host);
  const xs = o.series[0].values.map(v => v.x);
  const x = linear(o.xDomain[0], o.xDomain[1], m.l, width - m.r);
  const y = linear(o.yDomain[0], o.yDomain[1], H - m.b, m.t);

  for (const t of o.yTicks) {
    el('line', { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: 'viz-grid' }, svg);
    el('text', { x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end', class: 'viz-axis' }, svg).textContent = o.yFmt(t);
  }
  for (const t of o.xTicks) {
    el('text', { x: x(t), y: H - m.b + 18, 'text-anchor': 'middle', class: 'viz-axis' }, svg).textContent = t;
  }
  if (o.xLabel) el('text', { x: (m.l + width - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'viz-axis' }, svg).textContent = o.xLabel;
  if (o.yLabel) el('text', { x: m.l, y: 10, class: 'viz-axis' }, svg).textContent = o.yLabel;
  if (o.diagonal) el('line', { x1: x(o.xDomain[0]), y1: y(o.xDomain[0]), x2: x(o.xDomain[1]), y2: y(o.xDomain[1]), class: 'viz-ref' }, svg);
  for (const r of o.refs || []) {
    el('line', { x1: m.l, x2: width - m.r, y1: y(r.y), y2: y(r.y), class: 'viz-ref' }, svg);
    el('text', { x: m.l + 6, y: y(r.y) - 6, class: 'viz-ref-label' }, svg).textContent = r.label;
  }
  for (const k of o.markers || []) {
    if (k.x < o.xDomain[0] || k.x > o.xDomain[1]) continue;
    el('line', { x1: x(k.x), x2: x(k.x), y1: m.t, y2: H - m.b, class: 'viz-ref' }, svg);
    el('text', { x: x(k.x) + 5, y: m.t + 9, class: 'viz-ref-label' }, svg).textContent = k.label;
  }
  for (const s of o.series) {
    const d = s.values.map((v, i) => `${i ? 'L' : 'M'}${x(v.x).toFixed(1)},${y(v.y).toFixed(1)}`).join('');
    el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
    for (const v of (o.dots ? s.values : s.values.slice(-1))) {
      el('circle', { cx: x(v.x), cy: y(v.y), r: 4, fill: s.color, stroke: 'var(--surface)', 'stroke-width': 2 }, svg);
    }
    if (o.endLabels) {
      const end = s.values[s.values.length - 1];
      el('text', { x: x(end.x) + 9, y: y(end.y) + 4, class: 'viz-end' }, svg).textContent = (o.valueFmt || o.yFmt)(end.y);
    }
  }
  const hair = el('line', { y1: m.t, y2: H - m.b, class: 'viz-hair', visibility: 'hidden' }, svg);
  let at = xs.length - 1;

  function focusAt(i, cx, cy) {
    at = Math.max(0, Math.min(xs.length - 1, i));
    hair.setAttribute('x1', x(xs[at]));
    hair.setAttribute('x2', x(xs[at]));
    hair.setAttribute('visibility', 'visible');
    const box = svg.getBoundingClientRect();
    showTip(cx ?? box.left + x(xs[at]), cy ?? box.top + m.t, o.tipTitle(xs[at]),
      o.series.map(s => ({ color: s.color, value: (o.valueFmt || o.yFmt)(s.values[at].y), label: s.tipLabel ? s.tipLabel(s.values[at]) : s.name })));
  }
  svg.addEventListener('pointermove', e => {
    const px = e.clientX - svg.getBoundingClientRect().left;
    let best = 0;
    xs.forEach((v, i) => { if (Math.abs(x(v) - px) < Math.abs(x(xs[best]) - px)) best = i; });
    focusAt(best, e.clientX, e.clientY);
  });
  const clear = () => { hair.setAttribute('visibility', 'hidden'); hideTip(); };
  svg.addEventListener('pointerleave', clear);
  svg.addEventListener('blur', clear);
  svg.addEventListener('focus', () => focusAt(at));
  svg.addEventListener('keydown', e => {
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); focusAt(at + (e.key === 'ArrowRight' ? 1 : -1)); }
  });
}

// ---- Scatter: small multiples on one scale; the pointer only has to be closest.

function scatterPanel(host, width, o) {
  const H = Math.min(340, Math.max(260, width * 0.8));
  const m = { t: 12, r: 12, b: 38, l: 42 };
  const svg = el('svg', { width, height: H, role: 'img', 'aria-label': o.aria, class: 'viz-svg' }, host);
  const x = linear(-o.xMax, o.xMax, m.l, width - m.r);
  const y = linear(-o.yMax, o.yMax, H - m.b, m.t);
  for (const t of [-o.yMax, -o.yMax / 2, 0, o.yMax / 2, o.yMax]) {
    el('line', { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: t ? 'viz-grid' : 'viz-ref' }, svg);
    el('text', { x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end', class: 'viz-axis' }, svg).textContent = t;
  }
  for (const t of [-o.xMax, -o.xMax / 2, 0, o.xMax / 2, o.xMax]) {
    el('text', { x: x(t), y: H - m.b + 18, 'text-anchor': 'middle', class: 'viz-axis' }, svg).textContent = t;
  }
  el('text', { x: (m.l + width - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'viz-axis' }, svg).textContent = o.xLabel;
  el('line', { x1: x(-o.xMax), y1: y(-o.xMax), x2: x(o.xMax), y2: y(o.xMax), class: 'viz-ref' }, svg);
  const fit = corr(o.points.map(p => p.x), o.points.map(p => p.y));
  el('line', { x1: x(-o.xMax), y1: y(fit.intercept - fit.slope * o.xMax), x2: x(o.xMax), y2: y(fit.intercept + fit.slope * o.xMax), stroke: o.color, 'stroke-width': 2 }, svg);
  const marks = o.points.map(p => el('circle', {
    cx: x(p.x), cy: y(Math.max(-o.yMax, Math.min(o.yMax, p.y))), r: 4,
    fill: p.early ? 'var(--surface)' : o.color, stroke: p.early ? o.color : 'var(--surface)', 'stroke-width': p.early ? 1.5 : 1,
  }, svg));
  let lit = null;
  svg.addEventListener('pointermove', e => {
    const box = svg.getBoundingClientRect();
    const px = e.clientX - box.left, py = e.clientY - box.top;
    let best = -1, dist = 28 ** 2;
    o.points.forEach((p, i) => {
      const d = (x(p.x) - px) ** 2 + (y(p.y) - py) ** 2;
      if (d < dist) { dist = d; best = i; }
    });
    if (lit !== null) marks[lit].setAttribute('r', 4);
    lit = best >= 0 ? best : null;
    if (lit === null) { hideTip(); return; }
    marks[lit].setAttribute('r', 7);
    const p = o.points[lit];
    showTip(e.clientX, e.clientY, p.title, p.rows);
  });
  svg.addEventListener('pointerleave', () => { if (lit !== null) marks[lit].setAttribute('r', 4); lit = null; hideTip(); });
}

// ---- Rows of dots on one shared axis: forest plots, dumbbells and bars.
// A row: {label, group?, points: [{series, value, lo?, hi?, color, name}], note?, tip?}

function dotRows(host, width, o) {
  const wantLabel = o.labelWidth || 270;
  const wide = width >= wantLabel + (o.noteWidth || 0) + 220;
  const labelW = wide ? wantLabel : 0;
  const noteW = wide && o.noteWidth ? o.noteWidth : 0;
  const plotW = Math.max(120, width - labelW - noteW);
  const pad = 14;
  const x = linear(o.domain[0], o.domain[1], pad, plotW - pad);
  const rowH = o.rowHeight || 30;
  const wrap = html('div', 'dot-rows', undefined, host);
  wrap.style.setProperty('--label', `${labelW}px`);
  wrap.style.setProperty('--note', `${noteW}px`);
  let group = null;
  for (const row of o.rows) {
    if (row.group && row.group !== group) { group = row.group; html('div', 'dot-group', group, wrap); }
    const r = html('div', `dot-row ${wide ? '' : 'stacked'}`, undefined, wrap);
    r.tabIndex = 0;
    r.setAttribute('aria-label', `${row.label}: ${row.points.map(p => `${p.name} ${o.fmt(p.value)}`).join(', ')}`);
    html('div', 'dot-label', row.label, r);
    const svg = el('svg', { width: plotW, height: rowH, class: 'viz-svg', 'aria-hidden': 'true' }, r);
    for (const t of o.ticks) el('line', { x1: x(t), x2: x(t), y1: 0, y2: rowH, class: o.zero === t ? 'viz-ref' : 'viz-grid' }, svg);
    if (o.marker !== undefined) el('line', { x1: x(o.marker), x2: x(o.marker), y1: 0, y2: rowH, class: 'viz-marker' }, svg);
    const cy = rowH / 2;
    if (o.connect && row.points.length > 1) {
      const vals = row.points.map(p => x(p.value));
      el('line', { x1: Math.min(...vals), x2: Math.max(...vals), y1: cy, y2: cy, class: 'viz-connect' }, svg);
    }
    const spread = o.offset || 0;
    row.points.forEach((p, i) => {
      const py = cy + (row.points.length > 1 ? (i - (row.points.length - 1) / 2) * spread : 0);
      if (p.lo !== undefined && p.lo !== null) {
        el('line', { x1: x(Math.max(o.domain[0], p.lo)), x2: x(Math.min(o.domain[1], p.hi)), y1: py, y2: py, stroke: p.color, 'stroke-width': 2, 'stroke-linecap': 'round' }, svg);
      }
      if (o.bars) {
        const x0 = x(0), x1 = x(p.value), h = 14, rad = Math.min(4, Math.abs(x1 - x0));
        const dir = x1 >= x0 ? 1 : -1;
        el('path', { d: `M${x0},${py - h / 2}H${x1 - dir * rad}Q${x1},${py - h / 2} ${x1},${py - h / 2 + rad}V${py + h / 2 - rad}Q${x1},${py + h / 2} ${x1 - dir * rad},${py + h / 2}H${x0}Z`, fill: p.color }, svg);
        const hiX = p.hi !== undefined ? x(p.hi) : x1;
        el('text', { x: Math.max(hiX, x1) + 6, y: py + 4, class: 'viz-end' }, svg).textContent = o.fmt(p.value);
      } else {
        el('circle', { cx: x(p.value), cy: py, r: 5, fill: p.color, stroke: 'var(--surface)', 'stroke-width': 2 }, svg);
      }
    });
    if (noteW) {
      const note = html('div', 'dot-note', undefined, r);
      if (row.noteCells) row.noteCells.forEach(c => {
        const cell = html('span', c.cls || '', c.text, note);
        cell.title = c.title || '';
      });
    } else if (row.noteCells) {
      const note = html('div', 'dot-note narrow', undefined, r);
      row.noteCells.forEach(c => html('span', c.cls || '', c.text, note));
    }
    const tipRows = row.tip || row.points.map(p => ({
      color: p.color, value: o.fmt(p.value),
      label: p.lo !== undefined && p.lo !== null ? `${p.name}, range ${o.fmt(p.lo)} to ${o.fmt(p.hi)}` : p.name,
    }));
    r.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, row.label, tipRows));
    r.addEventListener('pointerleave', hideTip);
    r.addEventListener('focus', () => { const b = r.getBoundingClientRect(); showTip(b.left + labelW + plotW / 2, b.top, row.label, tipRows); });
    r.addEventListener('blur', hideTip);
  }
  const axisRow = html('div', `dot-axis ${wide ? '' : 'stacked'}`, undefined, wrap);
  const axis = el('svg', { width: plotW, height: 34, class: 'viz-svg', 'aria-hidden': 'true' }, axisRow);
  for (const t of o.ticks) el('text', { x: x(t), y: 14, 'text-anchor': 'middle', class: 'viz-axis' }, axis).textContent = o.tickFmt ? o.tickFmt(t) : t;
  if (o.axisLabel) el('text', { x: plotW / 2, y: 31, 'text-anchor': 'middle', class: 'viz-axis' }, axis).textContent = o.axisLabel;
}

// ---- Sections ----------------------------------------------------------------

function renderSeason(acc, compare) {
  const std = acc.season_to_date;
  const weeks = std.map(w => w.week);
  const xDomain = [Math.min(1, weeks[0]), Math.max(weeks[weeks.length - 1], 8)];
  const markers = [{ x: 3, label: 'Sign fix' }, { x: 10, label: 'v2' }];
  const xTicks = weeks.filter(w => w % 2 === 0 || w === 1);
  const latest = std[std.length - 1];

  $('viz-ats-title').textContent = `${pct(latest.ats.accuracy)} against the spread, just over break-even`;
  mount('viz-ats', {
    table: { columns: ['Week', 'Right', 'Graded', 'Rate'], rows: std.map(w => [`Week ${w.week}`, w.ats.correct, w.ats.graded, pct(w.ats.accuracy)]) },
    draw: (host, w) => lineChart(host, w, {
      series: [{ name: 'Spread picks', color: SERIES.model, values: std.map(s => ({ x: s.week, y: s.ats.accuracy, n: s.ats })), tipLabel: v => `${v.n.correct} of ${v.n.graded} right` }],
      xDomain, xTicks, yDomain: [0.4, 0.7], yTicks: [0.4, 0.5, 0.6, 0.7], yFmt: v => pct(v, 0), valueFmt: v => pct(v), xLabel: 'Week', endLabels: true,
      refs: [{ y: 0.524, label: 'Break-even 52.4%' }], markers, tipTitle: x => `Through Week ${x}`,
      aria: `Season-to-date spread record by week, now ${pct(latest.ats.accuracy)}.`,
    }),
  });

  $('viz-corr-title').textContent = `Vegas leads, and the gap narrowed once the bug was fixed`;
  mount('viz-corr', {
    legend: [{ name: 'Our model', color: SERIES.model }, { name: 'Vegas', color: SERIES.vegas }],
    table: { columns: ['Week', 'Our model', 'Vegas', 'Our avg miss', 'Vegas avg miss'], rows: std.map(w => [`Week ${w.week}`, num(w.model_corr), num(w.vegas_corr), num(w.model_mae, 1), num(w.vegas_mae, 1)]) },
    draw: (host, w) => lineChart(host, w, {
      series: [
        { name: 'Our model', color: SERIES.model, values: std.map(s => ({ x: s.week, y: s.model_corr })) },
        { name: 'Vegas', color: SERIES.vegas, values: std.map(s => ({ x: s.week, y: s.vegas_corr })) },
      ],
      xDomain, xTicks, yDomain: [0, 0.5], yTicks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], yFmt: v => num(v), xLabel: 'Week', endLabels: true,
      markers, tipTitle: x => `Through Week ${x}`, aria: `Season-to-date correlation with final margins, model ${num(latest.model_corr)}, Vegas ${num(latest.vegas_corr)}.`,
    }),
  });

  const games = acc.weeks.flatMap(w => w.games.filter(g => g.result !== null && g.model_margin_home !== null && g.spread_home !== null).map(g => ({ ...g, week: w.week })));
  const describe = (g, m) => {
    const side = v => v === 0 ? 'Even' : `${v > 0 ? g.home_team : g.away_team} by ${Math.abs(v).toFixed(1)}`;
    return [{ value: side(m), label: 'projected', color: null }, { value: side(g.result).replace('.0', ''), label: 'final', color: null }];
  };
  const toPoints = key => games.map(g => ({
    x: g[key], y: g.result, early: g.week <= 2,
    title: `Week ${g.week}: ${g.away_team} @ ${g.home_team}`, rows: describe(g, g[key]),
  }));
  const later = games.filter(g => g.week > 2);
  const rOf = (list, key) => corr(list.map(g => g[key]), list.map(g => g.result)).r;
  $('viz-scatter-title').textContent = `Since the fix, our margins correlate ${num(rOf(later, 'model_margin_home'))} with results, Vegas's ${num(rOf(later, 'spread_home'))}`;
  mount('viz-scatter', {
    legend: [{ name: 'Weeks 3 and later', color: 'var(--text-3)', dot: true }, { name: 'Weeks 1–2, before the sign fix', color: 'var(--text-3)', dot: true, hollow: true }],
    table: { columns: ['Game', 'Week', 'Our model', 'Vegas', 'Final (home)'], rows: games.map(g => [`${g.away_team} @ ${g.home_team}`, g.week, signed(g.model_margin_home, 1), signed(g.spread_home, 1), signed(g.result, 0)]) },
    draw: (host, w) => {
      const grid = html('div', 'small-multiples', undefined, host);
      const panelW = w >= 640 ? Math.floor((w - 24) / 2) : w;
      for (const p of [
        { name: 'Our model', key: 'model_margin_home', color: SERIES.model },
        { name: 'Vegas', key: 'spread_home', color: SERIES.vegas },
      ]) {
        const cell = html('div', '', undefined, grid);
        html('div', 'small-title', `${p.name}: r = ${num(rOf(games, p.key))} all season`, cell);
        scatterPanel(cell, panelW, {
          points: toPoints(p.key), color: p.color, xMax: 20, yMax: 40, xLabel: `${p.name}'s projected home margin`,
          aria: `${p.name} projected margin against actual margin for ${games.length} games.`,
        });
      }
    },
  });

  if (compare) {
    const bins = s => compare.calibration.find(c => c.series === s).bins;
    $('viz-calibration-sub').textContent = `Games grouped by how much the forecast favored a team, across ${compare.calibration_games.toLocaleString()} games from 2021 on, replayed with v2. On the diagonal, a 7-point favorite wins by 7 on average. 2021–2024 are the seasons v2 was fit on.`;
    mount('viz-calibration', {
      legend: [{ name: 'Our model (v2)', color: SERIES.model }, { name: 'Vegas', color: SERIES.vegas }],
      table: {
        columns: ['Forecast', 'Predicted', 'Actual', '95% range', 'Games'],
        rows: ['model', 'vegas'].flatMap(s => bins(s).map(b => [s === 'model' ? 'Our model' : 'Vegas', num(b.pred, 1), num(b.actual, 1), `${num(b.lo, 1)} to ${num(b.hi, 1)}`, b.n])),
      },
      draw: (host, w) => {
        const pts = s => bins(s).map(b => ({ x: b.pred, y: b.actual, n: b.n }));
        const model = pts('model'), vegas = pts('vegas');
        const X = Math.max(...model.map(p => p.x), ...vegas.map(p => p.x)) + 1;
        const m = { t: 14, r: 14, b: 38, l: 44 }, H = 280;
        const svg = el('svg', { width: w, height: H, class: 'viz-svg', role: 'img', 'aria-label': 'Predicted versus actual average margin, both forecasts near the diagonal.' }, host);
        const x = linear(0, X, m.l, w - m.r), y = linear(-2, 15, H - m.b, m.t);
        for (const t of [0, 5, 10, 15]) {
          el('line', { x1: m.l, x2: w - m.r, y1: y(t), y2: y(t), class: t ? 'viz-grid' : 'viz-ref' }, svg);
          el('text', { x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end', class: 'viz-axis' }, svg).textContent = t;
        }
        for (const t of [0, 3, 6, 9, 12]) el('text', { x: x(t), y: H - m.b + 18, 'text-anchor': 'middle', class: 'viz-axis' }, svg).textContent = t;
        el('text', { x: (m.l + w - m.r) / 2, y: H - 4, 'text-anchor': 'middle', class: 'viz-axis' }, svg).textContent = 'Predicted margin for the favored team';
        el('line', { x1: x(0), y1: y(0), x2: x(X), y2: y(X), class: 'viz-ref' }, svg);
        for (const [points, color, name] of [[model, SERIES.model, 'Our model'], [vegas, SERIES.vegas, 'Vegas']]) {
          el('path', { d: points.map((p, i) => `${i ? 'L' : 'M'}${x(p.x)},${y(p.y)}`).join(''), fill: 'none', stroke: color, 'stroke-width': 2 }, svg);
          for (const p of points) {
            el('circle', { cx: x(p.x), cy: y(p.y), r: 5, fill: color, stroke: 'var(--surface)', 'stroke-width': 2 }, svg);
            const hit = el('circle', { cx: x(p.x), cy: y(p.y), r: 14, fill: 'transparent' }, svg);
            hit.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, name, [
              { color, value: num(p.y, 1), label: `actual, predicted ${num(p.x, 1)}` }, { value: String(p.n), label: 'games' },
            ]));
            hit.addEventListener('pointerleave', hideTip);
          }
        }
      },
    });

    const fig = $('viz-compare');
    const box = html('div', 'table-scroll compare-table', undefined, fig);
    const t = html('table', 'data-table', undefined, box);
    const head = html('tr', '', undefined, html('thead', '', undefined, t));
    ['', 'Correlation', 'Avg miss (pts)', 'Winners', 'Vs. spread'].forEach((c, i) => html('th', i ? 'num-center' : 'left', c, head));
    const body = html('tbody', '', undefined, t);
    for (const [label, key] of [['2025 season', '2025'], [`2026, Weeks 1–${compare.holdouts['2026'].through_week}`, '2026']]) {
      const h = compare.holdouts[key];
      const group = html('td', 'wrap compare-group', `${label} (${h.v2.games} games)`, html('tr', '', undefined, body));
      group.colSpan = 5;
      for (const [name, c] of [['v1', h.v1], ['v2', h.v2], ['Vegas', h.vegas]]) {
        const tr = html('tr', name === 'v2' ? 'compare-v2' : '', undefined, body);
        html('td', '', name, tr);
        html('td', 'num-center', num(c.corr), tr);
        html('td', 'num-center', num(c.mae), tr);
        html('td', 'num-center', pct(c.winners), tr);
        html('td', 'num-center', c.ats === undefined ? '—' : pct(c.ats), tr);
      }
    }
    const d = compare.holdouts['2025'].v2_minus_v1_mse, d26 = compare.holdouts['2026'].v2_minus_v1_mse;
    html('p', 'viz-note', `v2 cut squared error by ${Math.abs(d.diff).toFixed(1)} on 2025 (95% range ${Math.abs(d.hi).toFixed(1)} to ${Math.abs(d.lo).toFixed(1)}) and by ${Math.abs(d26.diff).toFixed(1)} on 2026 so far (${num(-d26.hi, 1)} to ${num(-d26.lo, 1)}). It predicts better. It still doesn't beat the spread: neither version clears 52.4%. The idea to try these two terms came from a test that included those seasons, so the holdouts aren't perfectly clean; the constants never saw them.`, fig);
  }
}

const FACTOR_GROUPS = {
  off_loss: 'Motivation: last week', off_blowout_loss: 'Motivation: last week', off_blowout_win: 'Motivation: last week',
  lost_cause: 'Motivation: the playoff race', on_the_cusp: 'Motivation: the playoff race', playoff_safe: 'Motivation: the playoff race',
  off_bye: 'Schedule', rest_diff: 'Schedule', west_early: 'Schedule',
  revenge: 'Storylines', qb_downgrade: 'Personnel', qb_upgrade: 'Personnel',
};
const FACTOR_STATUS = {
  qb_downgrade: { text: 'In the model', cls: 'mini-tag flag' },
  lost_cause: { text: 'Context tag', cls: 'mini-tag' },
  on_the_cusp: { text: 'Context tag', cls: 'mini-tag' },
  playoff_safe: { text: 'Context tag', cls: 'mini-tag' },
  rest_diff: { text: 'Context tag', cls: 'mini-tag' },
  revenge: { text: 'Removed in v2', cls: 'mini-tag' },
};

function renderMotivation(factors) {
  const f = factors.factors;
  const coach = f.find(x => x.key === 'former_coach');
  $('motivation-intro').textContent = `We tested the situations fans and TV love to talk about on ${factors.market_games.toLocaleString()} games back to ${factors.market_seasons[0]}, using only what was known before kickoff. The model test asks whether a factor explains our misses (${factors.model_games.toLocaleString()} games since ${factors.model_seasons[0]}). The Vegas test asks whether it explains the line's misses. A factor can matter on the field and still be useless for picks if Vegas already prices it in. No factor got the side that has it to the 52.4% break-even by a meaningful margin.`;
  $('viz-forest-title').textContent = `Only a step down at QB clearly moves the needle, and Vegas already has it`;
  const order = ['off_loss', 'off_blowout_loss', 'off_blowout_win', 'lost_cause', 'on_the_cusp', 'playoff_safe', 'off_bye', 'rest_diff', 'west_early', 'revenge', 'qb_downgrade', 'qb_upgrade'];
  const rows = order.map(key => {
    const x = f.find(y => y.key === key);
    const status = FACTOR_STATUS[key] || { text: 'Not scored', cls: 'mini-tag' };
    return {
      group: FACTOR_GROUPS[key], label: x.label,
      points: [
        { name: 'Vs. our model', value: x.model.pts, lo: x.model.lo, hi: x.model.hi, color: SERIES.model },
        { name: 'Vs. Vegas', value: x.market.pts, lo: x.market.lo, hi: x.market.hi, color: SERIES.vegas },
      ],
      noteCells: [{ text: `${pct(x.covers.rate)} covered`, cls: 'dot-num', title: 'How often the side with this factor covered the spread' }, { text: status.text, cls: status.cls }],
      tip: [
        { color: SERIES.model, value: signed(x.model.pts), label: `vs. our model, range ${signed(x.model.lo)} to ${signed(x.model.hi)}` },
        { color: SERIES.vegas, value: signed(x.market.pts), label: `vs. Vegas, range ${signed(x.market.lo)} to ${signed(x.market.hi)}` },
        { value: pct(x.covers.rate), label: `covered, ${x.covers.n.toLocaleString()} games` },
      ],
    };
  });
  mount('viz-forest', {
    legend: [{ name: 'Vs. our model', color: SERIES.model, dot: true }, { name: 'Vs. Vegas', color: SERIES.vegas, dot: true }],
    note: `Facing last season's head coach isn't plotted: it came up ${coach.games} times in ${factors.market_seasons[1] - factors.market_seasons[0]} seasons, far too few to test.`,
    table: {
      columns: ['Factor', 'Games', 'Vs. our model', '95% range', 'Vs. Vegas', '95% range', 'Covered'],
      rows: f.filter(x => x.key !== 'backup_qb').map(x => [x.label, x.games.toLocaleString(), signed(x.model.pts), `${signed(x.model.lo)} to ${signed(x.model.hi)}`, signed(x.market.pts), `${signed(x.market.lo)} to ${signed(x.market.hi)}`, pct(x.covers.rate)]),
    },
    draw: (host, w) => dotRows(host, w, {
      rows, domain: [-4, 4], ticks: [-4, -3, -2, -1, 0, 1, 2, 3, 4], zero: 0, fmt: v => signed(v), offset: 11, rowHeight: 36, labelWidth: 300, noteWidth: 210,
      axisLabel: 'Points for the team with the factor',
    }),
  });
}

const VARIANT_ROWS = [
  ['recency 0.9', 'Weight recent games more (10% a week)', 'Tried and rejected'],
  ['recency 0.8', 'Weight recent games much more (20% a week)', 'Tried and rejected'],
  ['opponent-adjusted', 'Adjust for opponents faced', 'Tried and rejected'],
  ['rushing EPA x0.5', 'Count rushing EPA at half', 'Tried and rejected'],
  ['+ point differential', 'Add point differential', 'One at a time'],
  ['QB-swap x0.25', 'Shift the offense for a new starting QB', 'One at a time'],
  ['+ backup-QB flag', 'Flag any change from the usual QB', 'One at a time'],
  ['v2: differential + QB swap', 'Differential + new QB', 'Together'],
  ['v2 + QB downgrade flag', 'v2: differential + new QB + step down at QB', 'Together'],
  ['v2 + QB downgrade, rest, lost cause', 'v2 + rest and playoff race', 'Together'],
];

function renderRating(rating) {
  const v = Object.fromEntries(rating.variants.map(x => [x.name, x]));
  const live = v['v2 + QB downgrade flag'], base = v['live structure'];
  $('viz-rating-title').textContent = `v2 improves on v1 by about ${Math.abs(live.diff / base.mse * 100).toFixed(1)}%, the most any combination managed`;
  $('viz-rating-sub').textContent = `Change in squared prediction error against the v1 structure, ${rating.games.toLocaleString()} games from ${rating.seasons[0]} to ${rating.seasons[1]}, each season predicted by a fit on the others. Left of zero predicts better. Adding rest and the playoff race on top of v2 didn't help.`;
  mount('viz-rating', {
    table: {
      columns: ['Change', 'Vs. v1', '95% range', 'Correlation', 'Winners', 'Vs. spread'],
      rows: VARIANT_ROWS.map(([k, label]) => [label, signed(v[k].diff), `${signed(v[k].lo)} to ${signed(v[k].hi)}`, num(v[k].corr, 3), pct(v[k].winners), pct(v[k].ats)]),
    },
    draw: (host, w) => dotRows(host, w, {
      rows: VARIANT_ROWS.map(([k, label, group]) => ({
        group, label,
        points: [{ name: label, value: v[k].diff, lo: v[k].lo, hi: v[k].hi, color: k === 'v2 + QB downgrade flag' ? SERIES.model : SERIES.muted }],
      })),
      domain: [-5, 5], ticks: [-4, -2, 0, 2, 4], zero: 0, fmt: x => signed(x), rowHeight: 28, labelWidth: 320,
      axisLabel: 'Change in squared error (left is better)',
    }),
  });
}

const POSITIONS = ['QB', 'RB', 'WR', 'TE', 'K', 'DST'];
const WEIGHT_LABELS = {
  form: 'Recent form (per point)', team_total: 'Vegas team total (per point)', matchup: 'Opposing defense (per point)',
  targets: 'Targets per game', carries: 'Carries per game', questionable: 'Listed Questionable', doubtful: 'Listed Doubtful',
  out: 'Listed Out', reserve: 'On a reserve list', not_starter: 'Not the starter', intercept: 'Starting point',
};

function renderFantasy(lab, weeklyAcc) {
  const by = lab.draft.by_position;
  const beat = POSITIONS.reduce((n, p) => n + by[p].consensus_beat_ours, 0), seasons = POSITIONS.reduce((n, p) => n + by[p].seasons, 0);
  $('viz-draft-title').textContent = `Experts beat a stats-only draft board in ${beat} of ${seasons} position-seasons`;
  $('viz-draft-sub').textContent = `Rank correlation with that season's points per game, averaged over 2021–2025. Experts see camp battles, rookies and free agency. Last year's stats don't. That's why the 2027 board leans 75% on consensus.`;
  const series = [
    { key: '0.0', name: 'Our stats-only board', color: SERIES.model },
    { key: '0.75', name: '75/25 blend', color: SERIES.muted },
    { key: '1.0', name: 'FantasyPros consensus', color: SERIES.vegas },
  ];
  mount('viz-draft', {
    legend: series.map(s => ({ ...s, dot: true })),
    table: {
      columns: ['Position', 'Season', 'Ours', '75/25 blend', 'Consensus', 'Players'],
      rows: POSITIONS.flatMap(p => by[p].by_season.map(r => [p, r.season, num(r.ours), num(r.blend), num(r.consensus), r.n])),
    },
    draw: (host, w) => dotRows(host, w, {
      rows: POSITIONS.map(p => ({ label: p, points: series.map(s => ({ name: s.name, value: by[p].means[s.key], color: s.color })) })),
      domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: v => num(v), connect: true, labelWidth: 50, axisLabel: 'Rank correlation with results',
    }),
  });

  const ppr = lab.weekly.ppr;
  $('viz-gains-sub').textContent = `The gain in rank correlation from adding one input to recent form alone, ${ppr.position_weeks.toLocaleString()} position-weeks since 2021, each season predicted by a fit on the others. The whiskers are 95% ranges.`;
  const gains = [
    ['availability', "Who's expected to play (injury report, reserve lists, QB starter)"],
    ['vegas', 'Vegas team total'],
    ['usage', 'Recent targets and carries'],
    ['matchup', 'Opposing defense vs. the position'],
    ['team_qb', "A receiver's QB is out"],
    ['former_team', 'Facing a former team'],
  ];
  mount('viz-gains', {
    table: { columns: ['Input', 'Gain', '95% range'], rows: gains.map(([k, label]) => [label, signed(ppr.effects[k].diff, 3), `${signed(ppr.effects[k].lo, 3)} to ${signed(ppr.effects[k].hi, 3)}`]) },
    draw: (host, w) => dotRows(host, w, {
      rows: gains.map(([k, label]) => ({ label, points: [{ name: label, value: ppr.effects[k].diff, lo: ppr.effects[k].lo, hi: ppr.effects[k].hi, color: SERIES.model }] })),
      domain: [-0.03, 0.26], ticks: [0, 0.05, 0.1, 0.15, 0.2], zero: 0, fmt: v => signed(v, 3), bars: true, rowHeight: 30, labelWidth: 230, axisLabel: 'Gain in rank correlation',
    }),
  });

  const vc = ppr.vs_consensus;
  $('viz-weekly-title').textContent = `The fitted Start Score closes most of the gap to the experts`;
  $('viz-weekly-sub').textContent = `Rank correlation with that week's points, 2021–2026, players who sat out counted as zero. Hand-set was the score through Week 8: ${num(ppr.overall.start_score_v1)} overall. Fitted: ${num(ppr.overall.fitted)}. FantasyPros: ${num(ppr.overall.ecr)}. On 2026, a season the fit never saw, fitted ran ${num(ppr.season_2026.fitted)} and FantasyPros ${num(ppr.season_2026.ecr)}.`;
  const weekly = [
    { key: 'start_score_v1', name: 'Hand-set (through Week 8)', color: SERIES.muted },
    { key: 'fitted', name: 'Fitted Start Score', color: SERIES.model },
    { key: 'ecr', name: 'FantasyPros', color: SERIES.vegas },
  ];
  mount('viz-weekly', {
    legend: weekly.map(s => ({ ...s, dot: true })),
    table: {
      columns: ['Position', 'Recent PPG', 'Hand-set', 'Fitted', 'FantasyPros'],
      rows: POSITIONS.map(p => [p, num(ppr.by_position[p].recent_ppg), num(ppr.by_position[p].start_score_v1), num(ppr.by_position[p].fitted), num(ppr.by_position[p].ecr)]),
    },
    note: `Across every week FantasyPros still leads the fitted score by ${Math.abs(vc.fitted_minus_ecr.diff).toFixed(3)}. Vegas totals give us the edge at defenses. QB is flattered a little here, since these tests use who actually started while experts rank before that's always known.`,
    draw: (host, w) => dotRows(host, w, {
      rows: POSITIONS.map(p => ({ label: p, points: weekly.map(s => ({ name: s.name, value: ppr.by_position[p][s.key], color: s.color })) })),
      domain: [0, 0.7], ticks: [0, 0.2, 0.4, 0.6], fmt: v => num(v), connect: true, labelWidth: 50, axisLabel: 'Rank correlation with results',
    }),
  });

  if (weeklyAcc && weeklyAcc.available) {
    const season = weeklyAcc.season_by_format.ppr;
    const list = weeklyAcc.weeks.map(w => w.week);
    $('viz-receipts-title').textContent = `2026 receipts, Weeks ${list[0]}–${list[list.length - 1]}`;
    const rec = [
      { key: 'recent_ppg', name: 'Recent PPG', color: SERIES.muted },
      { key: 'start_score', name: 'Start Score as published', color: SERIES.model },
      { key: 'market', name: 'FantasyPros', color: SERIES.vegas },
    ];
    const positions = POSITIONS.filter(p => season[p]);
    mount('viz-receipts', {
      legend: rec.map(s => ({ ...s, dot: true })),
      table: {
        columns: ['Position', 'Recent PPG', 'Start Score', 'FantasyPros', 'Weeks'],
        rows: positions.map(p => [p, pct(season[p].recent_ppg), pct(season[p].start_score), pct(season[p].market), season[p].weeks]),
      },
      note: 'Weeks through 8 are the hand-set score, rebuilt from the git history of what the site showed. The fitted score is graded from Week 9 on.',
      draw: (host, w) => dotRows(host, w, {
        rows: positions.map(p => ({ label: p, points: rec.filter(s => season[p][s.key] !== undefined).map(s => ({ name: s.name, value: season[p][s.key], color: s.color })) })),
        domain: [0.4, 0.8], ticks: [0.4, 0.5, 0.6, 0.7, 0.8], zero: 0.5, fmt: v => pct(v), tickFmt: v => pct(v, 0), connect: true, labelWidth: 50, axisLabel: 'Head-to-head (50% is a coin flip)',
      }),
    });
  }

  const box = $('weights-table');
  const t = html('table', 'data-table', undefined, box);
  const head = html('tr', '', undefined, html('thead', '', undefined, t));
  ['Input', ...POSITIONS].forEach((c, i) => html('th', i ? 'num-center' : 'left', c, head));
  const body = html('tbody', '', undefined, t);
  for (const [key, label] of Object.entries(WEIGHT_LABELS)) {
    const tr = html('tr', '', undefined, body);
    html('td', 'wrap', label, tr);
    for (const p of POSITIONS) html('td', 'num-center', ppr.coefficients[p][key] === undefined ? '—' : signed(ppr.coefficients[p][key]), tr);
  }
}

// ---- More chart parts: labeled scatter, waterfall, sortable heat table.

function labeledScatter(host, width, o) {
  const H = o.height || Math.min(460, Math.max(300, width * 0.68));
  const m = { t: 14, r: 18, b: 44, l: 50 };
  const svg = el('svg', { width, height: H, class: 'viz-svg', role: 'img', 'aria-label': o.aria }, host);
  const x = linear(o.xDomain[0], o.xDomain[1], m.l, width - m.r);
  const y = linear(o.yDomain[0], o.yDomain[1], H - m.b, m.t);
  if (o.band) el('rect', { x: m.l, width: width - m.l - m.r, y: Math.min(y(o.band[0]), y(o.band[1])), height: Math.abs(y(o.band[1]) - y(o.band[0])), class: 'viz-band' }, svg);
  for (const t of o.yTicks) {
    el('line', { x1: m.l, x2: width - m.r, y1: y(t), y2: y(t), class: t === o.zeroY ? 'viz-ref' : 'viz-grid' }, svg);
    el('text', { x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end', class: 'viz-axis' }, svg).textContent = (o.yFmt || String)(t);
  }
  for (const t of o.xTicks) {
    if (t === o.zeroX) el('line', { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: 'viz-ref' }, svg);
    el('text', { x: x(t), y: H - m.b + 18, 'text-anchor': 'middle', class: 'viz-axis' }, svg).textContent = (o.xFmt || String)(t);
  }
  el('text', { x: (m.l + width - m.r) / 2, y: H - 6, 'text-anchor': 'middle', class: 'viz-axis' }, svg).textContent = o.xLabel;
  if (o.yLabel) el('text', { x: m.l, y: 9, class: 'viz-axis' }, svg).textContent = o.yLabel;
  for (const line of o.lines || []) el('line', { x1: x(line[0]), y1: y(line[1]), x2: x(line[2]), y2: y(line[3]), class: 'viz-ref' }, svg);
  for (const c of o.corners || []) el('text', { x: x(c.x), y: y(c.y), 'text-anchor': c.anchor || 'start', class: 'viz-ref-label' }, svg).textContent = c.text;

  // Quiet dots first, so the ones the chart is about sit on top.
  const ordered = [...o.points].sort((a, b) => Number(!!a.emph) - Number(!!b.emph));
  const marks = new Map(ordered.map(p => [p, el('circle', { cx: x(p.x), cy: y(p.y), r: p.emph ? 5 : 4, fill: p.color, stroke: 'var(--surface)', 'stroke-width': 2 }, svg)]));
  const taken = [];
  const free = (a, b, c, d) => !taken.some(t => a < t[2] && c > t[0] && b < t[3] && d > t[1]);
  for (const p of [...ordered].reverse()) {
    if (!p.label) continue;
    const w = p.label.length * 7, px = x(p.x), py = y(p.y);
    const spots = [[px + 7, py + 4, 'start'], [px - 7, py + 4, 'end'], [px, py - 8, 'middle'], [px, py + 15, 'middle']];
    for (const [tx, ty, anchor] of spots) {
      const left = anchor === 'start' ? tx : anchor === 'end' ? tx - w : tx - w / 2;
      if (left < m.l || left + w > width - 2 || ty - 10 < 2 || ty > H - m.b) continue;
      if (free(left, ty - 10, left + w, ty + 2)) {
        taken.push([left, ty - 10, left + w, ty + 2]);
        el('text', { x: tx, y: ty, 'text-anchor': anchor, class: p.emph ? 'viz-team strong' : 'viz-team' }, svg).textContent = p.label;
        break;
      }
    }
  }
  let lit = null;
  svg.addEventListener('pointermove', e => {
    const box = svg.getBoundingClientRect();
    const px = e.clientX - box.left, py = e.clientY - box.top;
    let best = null, dist = 26 ** 2;
    for (const p of o.points) {
      const d = (x(p.x) - px) ** 2 + (y(p.y) - py) ** 2;
      if (d < dist) { dist = d; best = p; }
    }
    if (lit) marks.get(lit).setAttribute('r', lit.emph ? 5 : 4);
    lit = best;
    if (!best) { hideTip(); return; }
    marks.get(best).setAttribute('r', 7);
    showTip(e.clientX, e.clientY, best.title, best.rows);
  });
  svg.addEventListener('pointerleave', () => { if (lit) marks.get(lit).setAttribute('r', lit.emph ? 5 : 4); lit = null; hideTip(); });
}

function waterfall(host, width, o) {
  const wide = width >= 560;
  const labelW = wide ? Math.min(300, width * 0.42) : 0;
  const plotW = width - labelW;
  let run = 0;
  const steps = o.steps.map(s => { const a = run; run += s.value; return { ...s, a, b: run }; });
  const edges = steps.flatMap(s => [s.a, s.b]).concat(o.marker ? [o.marker.value] : [], [0, o.total.value]);
  const lo = Math.min(...edges), hi = Math.max(...edges), pad = (hi - lo) * 0.16 + 1;
  const x = linear(lo - pad, hi + pad, 6, plotW - 6);
  const wrap = html('div', 'dot-rows', undefined, host);
  wrap.style.setProperty('--label', `${labelW}px`);
  wrap.style.setProperty('--note', '0px');
  const row = (label, draw, h = 28) => {
    const r = html('div', `dot-row ${wide ? '' : 'stacked'}`, undefined, wrap);
    html('div', 'dot-label', label, r);
    const svg = el('svg', { width: plotW, height: h, class: 'viz-svg', 'aria-hidden': 'true' }, r);
    if (lo < 0 || o.showZero) el('line', { x1: x(0), x2: x(0), y1: 0, y2: h, class: 'viz-ref' }, svg);
    if (o.marker) el('line', { x1: x(o.marker.value), x2: x(o.marker.value), y1: 0, y2: h, class: 'viz-marker' }, svg);
    draw(svg, h);
  };
  const tag = (svg, value, anchorAt, h, dir) => {
    el('text', { x: anchorAt + (dir >= 0 ? 6 : -6), y: h / 2 + 4, 'text-anchor': dir >= 0 ? 'start' : 'end', class: 'viz-end' }, svg).textContent = value;
  };
  for (const s of steps) {
    row(s.label, (svg, h) => {
      const a = x(Math.min(s.a, s.b)), b = x(Math.max(s.a, s.b));
      el('rect', { x: a, y: h / 2 - 7, width: Math.max(b - a, 1.5), height: 14, rx: 3, fill: s.value >= 0 ? SERIES.vegas : SERIES.model }, svg);
      const right = b + 46 < plotW;
      tag(svg, signed(s.value, 1), right ? b : a, h, right ? 1 : -1);
    });
  }
  row(o.total.label, (svg, h) => {
    const a = x(Math.min(0, o.total.value)), b = x(Math.max(0, o.total.value));
    el('rect', { x: a, y: h / 2 - 8, width: Math.max(b - a, 1.5), height: 16, rx: 3, fill: 'var(--text-2)' }, svg);
    const text = o.totalFmt ? o.totalFmt(o.total.value) : num(o.total.value, 1);
    const right = b + text.length * 8 + 10 < plotW;
    tag(svg, text, right ? b : a, h, right ? 1 : -1);
  }, 32);
  if (o.marker) {
    const key = html('div', 'viz-legend', undefined, host);
    const item = html('span', 'viz-legend-item', undefined, key);
    html('span', 'viz-key marker', undefined, item);
    html('span', '', o.marker.label, item);
  }
}

const VERDICT_COLOR = { better_than_record: SERIES.vegas, sell_high: SERIES.model };
const VERDICT_PILL = { for_real: 'up', better_than_record: 'up', sell_high: 'down', real_problems: 'down', middle: 'pos' };

function heatTable(host, teams) {
  const rank = v => {
    const td = document.createElement('td');
    td.className = 'heat';
    td.textContent = v;
    const share = Math.round(Math.abs(v - 16.5) / 15.5 * 48);
    td.style.background = `color-mix(in srgb, ${v <= 16 ? SERIES.vegas : SERIES.model} ${share}%, var(--surface))`;
    return td;
  };
  const cols = [
    { key: 'team', head: 'Team', get: t => t.team, text: true },
    { key: 'record', head: 'Record', get: t => t.margin_rank, show: t => t.record },
    { key: 'margin', head: 'Margin', get: t => -t.margin, show: t => signed(t.margin, 1) },
    { key: 'verdict', head: 'Verdict', get: t => t.forecast_rank - t.margin_rank, verdict: true },
    { key: 'off', head: 'Offense', group: 'o', get: t => t.offense_rank, heat: true },
    { key: 'pass', head: 'Pass', group: 'o', get: t => t.ranks.pass, heat: true },
    { key: 'rush', head: 'Run', group: 'o', get: t => t.ranks.rush, heat: true },
    { key: 'succ', head: 'Success', group: 'o', get: t => t.ranks.success, heat: true },
    { key: 'big', head: 'Big plays', group: 'o', get: t => t.ranks.explosive, heat: true },
    { key: 'def', head: 'Defense', group: 'd', get: t => t.defense_rank, heat: true },
    { key: 'dpass', head: 'Pass D', group: 'd', get: t => t.ranks.def_pass, heat: true },
    { key: 'drush', head: 'Run D', group: 'd', get: t => t.ranks.def_rush, heat: true },
    { key: 'hits', head: 'QB hits', group: 'd', get: t => t.ranks.hits, heat: true },
    { key: 'sacks', head: 'Sacks', group: 'd', get: t => t.ranks.sacks, heat: true },
    { key: 'take', head: 'Takeaways', group: 'd', get: t => t.ranks.takeaways, heat: true },
  ];
  const table = html('table', 'data-table heat-table', undefined, host);
  const head = html('tr', '', undefined, html('thead', '', undefined, table));
  const body = html('tbody', '', undefined, table);
  let sortBy = cols.find(c => c.key === 'margin'), dir = 1;

  function paint() {
    head.querySelectorAll('th').forEach((th, i) => th.setAttribute('aria-sort', cols[i] === sortBy ? (dir > 0 ? 'ascending' : 'descending') : 'none'));
    const sorted = [...teams].sort((a, b) => {
      const x = sortBy.get(a), y = sortBy.get(b);
      return (x > y ? 1 : x < y ? -1 : 0) * dir;
    });
    body.replaceChildren();
    for (const t of sorted) {
      const tr = html('tr', '', undefined, body);
      for (const c of cols) {
        if (c.heat) tr.appendChild(rank(c.get(t)));
        else if (c.verdict) html('span', `pill ${VERDICT_PILL[t.verdict]}`, t.verdict_label, html('td', 'left-cell', undefined, tr));
        else html('td', c.text ? 'team-cell' : 'num-center', c.show ? c.show(t) : c.get(t), tr);
      }
    }
  }
  cols.forEach(c => {
    const th = html('th', c.group === 'd' ? 'num-center unit-start' : 'num-center', undefined, head);
    const b = html('button', 'heat-sort', c.head, th);
    b.type = 'button';
    b.addEventListener('click', () => { if (sortBy === c) dir = -dir; else { sortBy = c; dir = 1; } paint(); });
  });
  paint();
}

// ---- The new sections.

function tile(box, t) {
  const a = html('a', 'stat-tile card feature-tile', undefined, box);
  a.href = t.href;
  html('div', 'stat-label', t.label, a);
  html('div', 'stat-value', t.value, a);
  html('div', 'stat-sub', t.sub, a);
  html('div', 'tile-more', t.more, a);
}

function renderFound(wind, profiles, stream, lab) {
  const box = $('feature-tiles');
  if (wind && wind.backtest) {
    const r = wind.backtest.forecast.rule, live = wind.record;
    tile(box, { href: '#wind', label: 'Wind Watch', value: pct(r.rate), sub: `Unders with 10+ mph forecast, ${r.n} games, 2022–26. Break-even is 52.4%. ${live.under + live.over ? `Live: ${live.under}–${live.over}.` : 'Live test starts Sunday.'}`, more: 'See the evidence' });
  }
  if (stream) tile(box, { href: '#streaming', label: 'Defense streaming', value: `${signed(stream.ours_minus_fp, 1)} pts`, sub: `A week over FantasyPros' top streamer: ${stream.ours.toFixed(1)} vs. ${stream.fantasypros.toFixed(1)} across ${stream.weeks} weeks.`, more: 'See the weeks' });
  if (profiles && profiles.history) {
    const c = profiles.history.carried;
    tile(box, { href: '#verdicts', label: 'Team verdicts', value: `${(c.bottom_half_defense.margin_later - c.bottom_half_offense.margin_later).toFixed(1)} pts`, sub: `A game: how much better winners carried by offense finish than winners carried by defense, from about the same margin.`, more: 'See all 32 teams' });
  }
  if (lab) {
    const o = lab.weekly.ppr.overall;
    tile(box, { href: '#start-score', label: 'Start Score', value: num(o.fitted), sub: `Rank correlation with weekly points, up from ${num(o.start_score_v1)} when it was hand-set. FantasyPros: ${num(o.ecr)}.`, more: 'See the math' });
  }
}

function renderWind(wind, week, lab) {
  if (!wind || !wind.backtest) return;
  const b = wind.backtest, fc = b.forecast;
  $('wind-intro').textContent = `Over/under totals don't fully adjust for wind that's already in the forecast. Each mph of forecast wind has meant ${Math.abs(fc.pts_per_mph).toFixed(2)} fewer points than the total (95% range ${Math.abs(fc.slope_hi).toFixed(2)} to ${Math.abs(fc.slope_lo).toFixed(2)}), and wind recorded at the game shows the same slope in both 2006–18 (${Math.abs(b.recorded['2006-2018'].pts_per_mph).toFixed(2)}) and 2019–26 (${Math.abs(b.recorded['2019-2026'].pts_per_mph).toFixed(2)}). The range on the rule's hit rate still reaches below break-even, and among the 23 rules in the edge hunt, luck alone matches this one about ${lab ? Math.round(lab.rules.luck['Wind 10+ mph'] * 100) : 19}% of the time. That's why it's running as a live test.`;
  const live = wind.record;
  $('wind-live-note').textContent = live.under + live.over ? `Live record: ${live.under}–${live.over}${live.push ? `–${live.push}` : ''} on the under` : 'First live grades arrive after Sunday.';
  $('viz-wind-buckets-sub').textContent = `Under rate by forecast wind, ${fc.games} open-air games, 2022–26, with 95% ranges. The line is the ${pct(b.break_even)} break-even. The rule was set at 10 mph, and the 15+ group is only ${fc.buckets[3].n} games.`;
  const bucketName = x => x.to_mph >= 99 ? `${x.from_mph}+ mph` : x.from_mph === 0 ? `Under ${x.to_mph} mph` : `${x.from_mph}–${x.to_mph} mph`;
  mount('viz-wind-buckets', {
    table: { columns: ['Forecast wind', 'Under rate', '95% range', 'Games'], rows: fc.buckets.map(x => [bucketName(x), pct(x.rate), `${pct(x.lo)} to ${pct(x.hi)}`, x.n]) },
    draw: (host, w) => dotRows(host, w, {
      rows: fc.buckets.map(x => ({ label: bucketName(x), points: [{ name: 'Under rate', value: x.rate, lo: x.lo, hi: x.hi, color: x.from_mph >= 10 ? SERIES.model : SERIES.muted }] })),
      domain: [0.2, 0.85], ticks: [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], zero: null, marker: b.break_even, fmt: v => pct(v), tickFmt: v => pct(v, 0), labelWidth: 96, rowHeight: 34, axisLabel: 'Games going under the total',
    }),
  });
  const seasons = Object.entries(fc.by_season).map(([y, x]) => ({ group: 'Forecast 10+ mph', label: y, x }));
  const eras = ['2006-2018', '2019-2026'].map(y => ({ group: 'Recorded 10+ mph', label: y.replace('-', '–'), x: b.recorded[y] }));
  mount('viz-wind-seasons', {
    table: { columns: ['Sample', 'Under rate', '95% range', 'Games'], rows: [...seasons, ...eras].map(r => [`${r.group}, ${r.label}`, pct(r.x.rate), `${pct(r.x.lo)} to ${pct(r.x.hi)}`, r.x.n]) },
    draw: (host, w) => dotRows(host, w, {
      rows: [...seasons, ...eras].map(r => ({ group: r.group, label: `${r.label} (${r.x.n})`, points: [{ name: 'Under rate', value: r.x.rate, lo: r.x.lo, hi: r.x.hi, color: r.x.rate >= b.break_even ? SERIES.model : SERIES.muted }] })),
      domain: [0.15, 0.95], ticks: [0.2, 0.4, 0.6, 0.8], marker: b.break_even, fmt: v => pct(v), tickFmt: v => pct(v, 0), labelWidth: 96, rowHeight: 26, axisLabel: 'Games going under the total',
    }),
  });
  const games = week && week.games ? week.games : [];
  if (games.length) {
    const spots = games.filter(g => g.spot).length;
    $('viz-wind-week-title').textContent = `Week ${week.week}: ${spots} of ${games.length} open-air games clear ${wind.rule_mph} mph`;
    const sorted = [...games].sort((a, c) => c.forecast_mph - a.forecast_mph);
    mount('viz-wind-week', {
      table: { columns: ['Game', 'Forecast mph', 'Peak gust', 'Total', 'Spot'], rows: sorted.map(g => [`${g.away_team} @ ${g.home_team}`, num(g.forecast_mph, 1), g.gust_mph === null ? '—' : num(g.gust_mph, 1), g.total ?? '—', g.spot ? 'Under' : '']) },
      draw: (host, w) => dotRows(host, w, {
        rows: sorted.map(g => ({
          label: `${g.away_team} @ ${g.home_team}`,
          points: [{ name: 'Forecast', value: g.forecast_mph, lo: g.forecast_mph, hi: Math.max(g.gust_mph ?? g.forecast_mph, g.forecast_mph), color: g.spot ? SERIES.model : SERIES.muted }],
          tip: [{ color: g.spot ? SERIES.model : SERIES.muted, value: `${num(g.forecast_mph, 1)} mph`, label: 'forecast' }, { value: `${num(g.gust_mph ?? 0, 0)} mph`, label: 'peak gust' }, { value: String(g.total ?? '—'), label: 'total' }],
        })),
        domain: [0, 36], ticks: [0, 5, 10, 15, 20, 25, 30, 35], marker: wind.rule_mph, fmt: v => `${num(v, 1)} mph`, tickFmt: v => v, labelWidth: 96, rowHeight: 26, axisLabel: 'Miles per hour. Dot: forecast. Whisker: peak gust. Line: the rule.',
      }),
    });
  } else {
    $('viz-wind-week-title').textContent = 'This week\'s forecasts appear once the week\'s lines are posted.';
  }
}

function renderTeams(profiles) {
  if (!profiles || !profiles.available) return;
  const h = profiles.history, teams = Object.entries(profiles.teams).map(([team, t]) => ({ team, ...t }));
  $('verdicts-intro').textContent = `Every game card gives both teams a verdict: their scoreboard rank against a forecast of their margin from here on, using point margin, this season's EPA and last season's. The history below comes from ${h.checkpoints.toLocaleString()} team checkpoints across ${h.seasons[0]}–${h.seasons[1]}, each season held out of the fit. The forecast is only slightly better than margin alone, so read a verdict as how similar teams finished, not as a promise.`;
  $('verdicts-note').textContent = `Live, through Week ${profiles.through_week}`;
  const tipRows = t => [{ value: `#${t.margin_rank}`, label: `by margin (${signed(t.margin, 1)} a game)` }, { value: `#${t.forecast_rank}`, label: `by forecast (${signed(t.forecast, 1)})` }, { value: t.verdict_label, label: t.record }];
  mount('viz-team-forecast', {
    legend: [{ name: 'Better than the record', color: SERIES.vegas, dot: true }, { name: 'Regression watch', color: SERIES.model, dot: true }, { name: 'Everyone else', color: SERIES.muted, dot: true }],
    table: { columns: ['Team', 'Margin rank', 'Forecast rank', 'Verdict'], rows: [...teams].sort((a, b) => a.margin_rank - b.margin_rank).map(t => [t.team, t.margin_rank, t.forecast_rank, t.verdict_label]) },
    draw: (host, w) => labeledScatter(host, w, {
      points: teams.map(t => ({ x: t.margin_rank, y: t.forecast_rank, label: t.team, emph: !!VERDICT_COLOR[t.verdict], color: VERDICT_COLOR[t.verdict] || SERIES.muted, title: t.team, rows: tipRows(t) })),
      xDomain: [0, 33], yDomain: [33, 0], xTicks: [1, 8, 16, 24, 32], yTicks: [1, 8, 16, 24, 32], xFmt: v => `#${v}`, yFmt: v => `#${v}`, xLabel: 'Rank by point margin so far', yLabel: 'Forecast rank',
      lines: [[0, 0, 33, 33], [0, 6, 27, 33], [6, 0, 33, 27]],
      corners: [{ x: 32.5, y: 2, anchor: 'end', text: 'Better than the record' }, { x: 0.8, y: 31.5, text: 'Regression watch' }],
      aria: 'Each team\'s rank by point margin against its rank by forecast.',
    }),
  });
  const xs = teams.map(t => t.offense_epa), ys = teams.map(t => t.defense_epa);
  const span = v => Math.ceil(Math.max(...v.map(Math.abs)) * 20 + 1) / 20;
  const sx = span(xs), sy = span(ys);
  mount('viz-team-build', {
    legend: [{ name: 'Better than the record', color: SERIES.vegas, dot: true }, { name: 'Regression watch', color: SERIES.model, dot: true }, { name: 'Everyone else', color: SERIES.muted, dot: true }],
    table: { columns: ['Team', 'Offense EPA/play', 'Rank', 'Defense EPA/play allowed', 'Rank'], rows: [...teams].sort((a, b) => a.offense_rank - b.offense_rank).map(t => [t.team, signed(t.offense_epa, 3), t.offense_rank, signed(t.defense_epa, 3), t.defense_rank]) },
    draw: (host, w) => labeledScatter(host, w, {
      points: teams.map(t => ({ x: t.offense_epa, y: t.defense_epa, label: t.team, emph: !!VERDICT_COLOR[t.verdict], color: VERDICT_COLOR[t.verdict] || SERIES.muted, title: t.team, rows: [{ value: signed(t.offense_epa, 3), label: `offense, #${t.offense_rank}` }, { value: signed(t.defense_epa, 3), label: `defense allowed, #${t.defense_rank}` }] })),
      xDomain: [-sx, sx], yDomain: [sy, -sy], xTicks: [-0.2, -0.1, 0, 0.1, 0.2].filter(t => Math.abs(t) <= sx), yTicks: [-0.2, -0.1, 0, 0.1, 0.2].filter(t => Math.abs(t) <= sy), zeroX: 0, zeroY: 0,
      xFmt: v => signed(v, 1), yFmt: v => signed(v, 1), xLabel: 'Offensive EPA per play', yLabel: 'Defensive EPA per play allowed',
      aria: 'Offensive and defensive EPA per play for all 32 teams.',
    }),
  });
  heatTable($('team-heat'), teams);

  const c = h.carried, names = { bottom_half_offense: 'Bottom-half offense', bottom_half_defense: 'Bottom-half defense', both_top_half: 'Top half in both' };
  $('viz-carried-sub').textContent = `Top-10 teams by margin, sorted by the unit that isn't pulling its weight. The difference between the first two is ${(c.bottom_half_defense.margin_later - c.bottom_half_offense.margin_later).toFixed(1)} points a game (95% range 0.1 to 7.5), in the same direction in 6 of 7 seasons. It's borderline on its own, but it lines up with what repeats below.`;
  mount('viz-carried', {
    legend: [{ name: 'So far', color: SERIES.muted, dot: true }, { name: 'Rest of the season', color: SERIES.model, dot: true }],
    table: { columns: ['Winners with a', 'Team-seasons', 'Margin so far', 'Rest of season', '95% range', 'Rank so far', 'Rank later'], rows: Object.keys(names).map(k => [names[k], c[k].team_seasons, signed(c[k].margin_so_far, 1), signed(c[k].margin_later, 1), `${signed(c[k].later_lo, 1)} to ${signed(c[k].later_hi, 1)}`, c[k].rank_so_far, c[k].rank_later]) },
    draw: (host, w) => dotRows(host, w, {
      rows: Object.keys(names).map(k => ({ label: names[k], points: [{ name: 'So far', value: c[k].margin_so_far, color: SERIES.muted }, { name: 'Rest of the season', value: c[k].margin_later, lo: c[k].later_lo, hi: c[k].later_hi, color: SERIES.model }] })),
      domain: [-2, 10], ticks: [0, 2, 4, 6, 8, 10], zero: 0, fmt: v => signed(v, 1), tickFmt: v => signed(v, 0), offset: 10, rowHeight: 40, labelWidth: 140, axisLabel: 'Point margin per game',
    }),
  });

  const v = h.verdicts, order = ['for_real', 'better_than_record', 'middle', 'sell_high', 'real_problems'];
  const labels = { for_real: 'For real', better_than_record: 'Better than the record', middle: 'About what it looks like', sell_high: 'Regression watch', real_problems: 'Real problems' };
  $('viz-verdict-history-sub').textContent = 'Average margin so far and over the rest of the season for teams given each verdict. Everything drifts toward average, so the test is the table\'s last column: the gap against a forecast from margin alone. Each points the right way, and none is conclusive by itself.';
  mount('viz-verdict-history', {
    legend: [{ name: 'So far', color: SERIES.muted, dot: true }, { name: 'Rest of the season', color: SERIES.model, dot: true }],
    table: { columns: ['Verdict', 'Checkpoints', 'Margin so far', 'Rest of season', 'Top 10 later', 'Bottom 10 later', 'Vs. margin alone'], rows: order.map(k => [labels[k], v[k].n, signed(v[k].margin_so_far, 1), signed(v[k].margin_later, 1), pct(v[k].top10_later, 0), pct(v[k].bottom10_later, 0), `${signed(v[k].vs_margin_only, 2)} (${signed(v[k].vs_margin_only_lo, 1)} to ${signed(v[k].vs_margin_only_hi, 1)})`]) },
    draw: (host, w) => dotRows(host, w, {
      rows: order.map(k => ({ label: labels[k], points: [{ name: 'So far', value: v[k].margin_so_far, color: SERIES.muted }, { name: 'Rest of the season', value: v[k].margin_later, color: SERIES.model }] })),
      domain: [-10, 10], ticks: [-10, -5, 0, 5, 10], zero: 0, fmt: x => signed(x, 1), tickFmt: x => signed(x, 0), connect: true, labelWidth: 170, axisLabel: 'Point margin per game',
    }),
  });

  let period = profiles.period || 'mid';
  const periodNames = { early: 'Weeks 4–6', mid: 'Weeks 8–10', late: 'Weeks 12–14' };
  const holdRows = () => h.hold_rates.offense[period].map((row, i) => ({ row, d: h.hold_rates.defense[period][i] }));
  const redrawHold = mount('viz-hold', {
    legend: [{ name: 'Offense', color: SERIES.model, dot: true }, { name: 'Defense', color: SERIES.vegas, dot: true }],
    table: { columns: ['Rank now', 'Weeks', 'Offense top 10 later', 'Defense top 10 later', 'Checkpoints'], rows: Object.keys(periodNames).flatMap(p => h.hold_rates.offense[p].map((row, i) => [`#${row.lo}–${row.hi}`, periodNames[p], pct(row.top10, 0), pct(h.hold_rates.defense[p][i].top10, 0), row.n])) },
    draw: (host, w) => dotRows(host, w, {
      rows: holdRows().map(({ row, d }) => ({ label: `#${row.lo}–${row.hi}`, points: [{ name: 'Offense', value: row.top10, color: SERIES.model }, { name: 'Defense', value: d.top10, color: SERIES.vegas }] })),
      domain: [0, 0.8], ticks: [0, 0.2, 0.4, 0.6, 0.8], fmt: x => pct(x, 0), tickFmt: x => pct(x, 0), connect: true, labelWidth: 70, axisLabel: 'Finishes top 10 the rest of the way',
    }),
  });
  const controls = $('hold-periods');
  for (const p of Object.keys(periodNames)) {
    const b = html('button', `toggle-btn ${p === period ? 'active' : ''}`, periodNames[p], controls);
    b.type = 'button';
    b.addEventListener('click', () => {
      period = p;
      controls.querySelectorAll('button').forEach(x => x.classList.toggle('active', x === b));
      redrawHold();
    });
  }

  const s = h.splits, storyNames = {
    'defense with top-10 QB hit rate': 'Defense with a top-10 QB hit rate',
    'defense leaning on takeaways': 'Defense leaning on takeaways',
    'defense top 10 without takeaways': 'Defense still top 10 without takeaways',
    'offense led by the pass': 'Offense led by the pass',
    'offense top 10 in both pass and run': 'Offense top 10 passing and running',
  };
  mount('viz-stories', {
    table: { columns: ['Story', 'With', 'Without', 'Difference', '95% range'], rows: Object.keys(storyNames).map(k => [storyNames[k], pct(s[k].with, 0), pct(s[k].without, 0), `${signed((s[k].with - s[k].without) * 100, 0)} pts`, `${signed(s[k].diff_lo * 100, 0)} to ${signed(s[k].diff_hi * 100, 0)}`]) },
    draw: (host, w) => dotRows(host, w, {
      rows: Object.keys(storyNames).map(k => ({ label: storyNames[k], points: [{ name: 'Change in odds', value: s[k].with - s[k].without, lo: s[k].diff_lo, hi: s[k].diff_hi, color: SERIES.model }] })),
      domain: [-0.3, 0.3], ticks: [-0.2, -0.1, 0, 0.1, 0.2], zero: 0, fmt: x => `${signed(x * 100, 0)} pts`, tickFmt: x => `${signed(x * 100, 0)}`, labelWidth: 210, rowHeight: 34, axisLabel: 'Percentage points of staying top 10',
    }),
  });
}

function renderStreaming(rankings, stream) {
  if (stream) {
    const ahead = stream.by_season.filter(y => y.ours > y.fantasypros).length;
    $('streaming-intro').textContent = `Take the defenses rostered in ${stream.max_rostered.toFixed(0)}% of leagues or fewer, and start the one facing the lowest Vegas team total. Over ${stream.weeks} weeks that's averaged ${stream.ours.toFixed(1)} fantasy points, against ${stream.fantasypros.toFixed(1)} for FantasyPros' top pick from the same group and ${stream.pool_average.toFixed(1)} for the average defense in it. The gain over FantasyPros is ${signed(stream.ours_minus_fp, 1)} points a week (95% range ${signed(stream.lo, 1)} to ${signed(stream.hi, 1)}), and the two make the same pick ${pct(stream.same_pick, 0)} of the time.`;
    $('viz-stream-seasons-title').textContent = `Ahead of the experts' top streamer in ${ahead} of ${stream.by_season.length} seasons`;
    const series = [{ key: 'pool_average', name: 'Average available defense', color: SERIES.muted }, { key: 'fantasypros', name: "FantasyPros' top pick", color: SERIES.vegas }, { key: 'ours', name: 'Lowest opponent total', color: SERIES.model }];
    mount('viz-stream-seasons', {
      legend: series.map(s => ({ ...s, dot: true })),
      table: { columns: ['Season', 'Weeks', 'Lowest opponent total', "FantasyPros' top pick", 'Average available'], rows: stream.by_season.map(y => [y.season, y.weeks, num(y.ours, 1), num(y.fantasypros, 1), num(y.pool_average, 1)]) },
      draw: (host, w) => dotRows(host, w, {
        rows: stream.by_season.map(y => ({ label: String(y.season), points: series.map(s => ({ name: s.name, value: y[s.key], color: s.color })) })),
        domain: [4, 12], ticks: [4, 6, 8, 10, 12], fmt: x => num(x, 1), connect: true, labelWidth: 56, rowHeight: 32, axisLabel: 'Fantasy points per week',
      }),
    });
  }
  const r = rankings && rankings.available ? rankings : null;
  if (!r) return;
  const pool = r.ppr.rankings_by_position.DST.filter(d => d.rostered_pct !== null && d.rostered_pct !== undefined);
  const picks = new Set(r.defense_streaming.picks.map(p => p.team));
  $('streaming-note').textContent = `Week ${r.defense_streaming.week} picks: ${r.defense_streaming.picks.map(p => p.team).join(', ')}`;
  $('viz-stream-map-title').textContent = `Week ${r.defense_streaming.week}: where the available defenses are`;
  mount('viz-stream-map', {
    legend: [{ name: "This week's picks", color: SERIES.model, dot: true }, { name: 'Other defenses', color: SERIES.muted, dot: true }],
    table: { columns: ['Defense', 'Start Score', 'Rostered', 'FP rank', 'Opponent'], rows: pool.map(d => [d.team, num(d.start_score, 1), `${Math.round(d.rostered_pct)}%`, d.market_rank ? Math.round(d.market_rank) : '—', d.opponent]) },
    draw: (host, w) => labeledScatter(host, w, {
      points: pool.map(d => ({ x: d.start_score, y: d.rostered_pct, label: picks.has(d.team) ? d.team : '', emph: picks.has(d.team), color: picks.has(d.team) ? SERIES.model : SERIES.muted, title: `${d.team} vs. ${d.opponent}`, rows: [{ value: num(d.start_score, 1), label: 'Start Score' }, { value: `${Math.round(d.rostered_pct)}%`, label: 'rostered' }, { value: d.market_rank ? `#${Math.round(d.market_rank)}` : '—', label: 'FantasyPros' }] })),
      xDomain: [Math.floor(Math.min(...pool.map(d => d.start_score))) - 0.5, Math.ceil(Math.max(...pool.map(d => d.start_score))) + 0.5], yDomain: [0, 100],
      xTicks: Array.from({ length: 8 }, (_, i) => 3 + i), yTicks: [0, 25, 50, 75, 100], band: [0, 50], yFmt: v => `${v}%`, xLabel: 'Start Score (projected points)', yLabel: 'Rostered',
      aria: 'Start Score against roster share for every defense this week.',
    }),
  });
}

const INPUT_LABELS = {
  form: 'Recent form', team_total: 'Vegas team total', matchup: 'Opposing defense', targets: 'Targets per game', carries: 'Carries per game',
  questionable: 'Listed Questionable', doubtful: 'Listed Doubtful', out: 'Listed Out', reserve: 'On a reserve list', not_starter: 'Not the starter',
};

function renderScoreExplorer(data) {
  if (!data) return;
  $('viz-score-explorer-sub').textContent = `Week ${data.week}, PPR. Each input times its fitted weight, added up from the position's starting point. The Start Score is a projection of the week's fantasy points.`;
  const pos = $('score-position'), player = $('score-player');
  for (const p of Object.keys(data.players)) html('option', '', p, pos).value = p;
  let plot = null, width = 0;
  const current = () => data.players[pos.value][Number(player.value) || 0];
  const fill = () => {
    player.replaceChildren();
    data.players[pos.value].forEach((x, i) => { html('option', '', `${i + 1}. ${x.name} (${x.team} vs. ${x.opponent})`, player).value = i; });
  };
  const draw = (host, w) => {
    plot = host; width = w;
    const x = current(), weights = data.weights[pos.value];
    const steps = [{ label: 'Starting point', value: weights.intercept }];
    for (const [k, v] of Object.entries(x.inputs)) {
      const unit = ['questionable', 'doubtful', 'out', 'reserve', 'not_starter'].includes(k) ? '' : ` (${num(v, 1)} × ${num(weights[k], 2)})`;
      steps.push({ label: `${INPUT_LABELS[k] || k}${unit}`, value: v * weights[k] });
    }
    waterfall(host, w, { steps, total: { label: `${x.name}: Start Score`, value: x.score }, totalFmt: v => num(v, 1), showZero: true });
  };
  mount('viz-score-explorer', { draw });
  const refresh = () => { if (plot) { plot.replaceChildren(); draw(plot, width); } };
  pos.value = 'WR';
  fill();
  pos.addEventListener('change', () => { fill(); refresh(); });
  player.addEventListener('change', refresh);
  refresh();
}

const BREAKDOWN_TEXT = {
  offense: r => `${r.team} offense vs. ${r.opp} defense`, point_diff: () => 'Season point differential', new_qb: r => `${r.team} new starting QB`, home_field: () => 'Home field',
  qb_out: r => `${r.team} starting QB out`, revenge: r => `${r.team} revenge game`, former_coach: r => `${r.team} faces former coach`,
};

function renderGameModel(board, compare, acc) {
  if (compare) {
    const k = compare.constants, f = $('model-formula');
    const lines = [
      ['margin', `${k.MARGIN_SCALE} × [(home off − away off) + (away def − home def)] + ${k.PD_WEIGHT} × (home diff − away diff) + ${k.HOME_FIELD_ADJ} home field − ${k.QB_OUT_PENALTY} step down at QB`],
      ['off', `w × this season + (1 − w) × (${k.RHO_OFF} × last season + ${k.QB_SWAP_SHARE} × QB gap)`],
      ['def', `w × this season + (1 − w) × ${k.RHO_DEF} × last season`],
      ['diff', `g × this season + (1 − g) × ${k.RHO_PD} × last season`],
      ['w, g', `plays / (plays + ${k.K_PLAYS}), games / (games + ${k.K_GAMES})`],
    ];
    for (const [a, b] of lines) {
      const row = html('div', 'formula-row', undefined, f);
      html('span', 'formula-name', a, row);
      html('span', 'formula-eq', '=', row);
      html('span', 'formula-body', b, row);
    }
    html('p', 'viz-note', `Off and def are EPA per play against league average, where a higher def number means a worse defense. Diff is point differential per game. QB gap is the new starter's EPA per dropback over his last two seasons, shrunk toward replacement level, minus last season's primary QB's. Constants are fit on ${compare.fit_seasons[0]}–${compare.fit_seasons[compare.fit_seasons.length - 1]}, by least squares on final margins.`, f);
    const vs = compare.vs_vegas;
    $('viz-gaps-sub').textContent = `v2 replayed on 2021–26: spread picks by how far the model's margin sat from the line. Nothing clears break-even at any size, and when the model and Vegas name different winners, the model is right ${pct(vs.different_winner.rate, 0)} of the time (${vs.different_winner.n} games).`;
    const gapName = x => x.gap_to >= 99 ? `${x.gap_from}+ points` : `${x.gap_from}–${x.gap_to} points`;
    mount('viz-gaps', {
      table: { columns: ['Gap from Vegas', 'Hit rate', '95% range', 'Games'], rows: vs.gaps.map(x => [gapName(x), pct(x.rate), `${pct(x.lo)} to ${pct(x.hi)}`, x.n]) },
      draw: (host, w) => dotRows(host, w, {
        rows: vs.gaps.map(x => ({ label: gapName(x), points: [{ name: 'Hit rate', value: x.rate, lo: x.lo, hi: x.hi, color: SERIES.model }] })),
        domain: [0.3, 0.75], ticks: [0.3, 0.4, 0.5, 0.6, 0.7], marker: 0.524, fmt: v => pct(v), tickFmt: v => pct(v, 0), labelWidth: 96, rowHeight: 32, axisLabel: 'Spread picks right. Line: break-even.',
      }),
    });
    $('viz-blend-sub').textContent = `Squared error on 2025–26, seasons the fit never saw, as the model's share of a blend grows. A regression of results on both gives Vegas a weight of ${num(vs.weight_all.vegas)} and the model ${num(vs.weight_all.model)} (95% range ${num(vs.weight_all.model_lo)} to ${num(vs.weight_all.model_hi)}) on 2021–26.`;
    mount('viz-blend', {
      table: { columns: ['Model share', 'Squared error', 'Average miss'], rows: vs.blend.map(x => [pct(x.model_share, 0), num(x.squared_error), num(x.avg_miss, 3)]) },
      draw: (host, w) => lineChart(host, w, {
        series: [{ name: 'Squared error', color: SERIES.model, values: vs.blend.map(x => ({ x: x.model_share * 100, y: x.squared_error })) }],
        xDomain: [0, 100], xTicks: [0, 25, 50, 75, 100], yDomain: [145, 170], yTicks: [145, 150, 155, 160, 165, 170], yFmt: v => String(v), valueFmt: v => num(v, 1), dots: true, endLabels: true, xLabel: "Model's share of the blend (%)",
        tipTitle: x => `${x}% model`, aria: 'Squared error rises as more of the model is mixed into the Vegas line.',
      }),
    });
  }
  if (board && board.games && board.games.length) {
    const select = $('game-select');
    const games = board.games.filter(g => g.margin_breakdown);
    games.forEach((g, i) => { html('option', '', `${g.away_team} @ ${g.home_team}`, select).value = i; });
    let plot = null, width = 0;
    const draw = (host, w) => {
      plot = host; width = w;
      const g = games[Number(select.value) || 0];
      const steps = g.margin_breakdown.map(r => ({ label: (BREAKDOWN_TEXT[r.kind] || (() => r.kind))(r), value: r.pts_home }));
      const spread = g.market ? g.market.spread_home : null;
      const side = v => v === 0 ? 'Even' : `${v > 0 ? g.home_team : g.away_team} by ${num(Math.abs(v), 1)}`;
      waterfall(host, w, {
        steps, total: { label: 'Model margin', value: g.model_margin_home }, totalFmt: side, showZero: true,
        marker: spread === null ? null : { label: `Vegas: ${side(spread)}. Right of zero favors ${g.home_team}, left favors ${g.away_team}.`, value: spread },
      });
    };
    mount('viz-game-explorer', { draw });
    select.addEventListener('change', () => { if (plot) { plot.replaceChildren(); draw(plot, width); } });
  }
  if (acc) {
    const latest = acc.season_to_date[acc.season_to_date.length - 1], fixed = acc.by_model.v1, box = $('stat-tiles');
    const plain = t => {
      const d = html('div', 'stat-tile card', undefined, box);
      html('div', 'stat-label', t.label, d);
      html('div', 'stat-value', t.value, d);
      html('div', 'stat-sub', t.sub, d);
    };
    plain({ label: 'Spread picks', value: `${acc.ats.correct}–${acc.ats.graded - acc.ats.correct}`, sub: `${pct(acc.ats.accuracy)}, break-even is 52.4%` });
    plain({ label: 'Margin correlation', value: num(latest.model_corr), sub: `Vegas ${num(latest.vegas_corr)}. Since the Week 3 fix: ${num(fixed.model_corr)} vs. ${num(fixed.vegas_corr)}` });
    plain({ label: 'Picked the winner', value: pct(acc.outright.accuracy), sub: `The Vegas favorite won ${pct(latest.vegas_winners)}` });
    plain({ label: 'Games graded', value: String(latest.games), sub: 'Every one frozen before kickoff' });
    $('data-through').textContent = `Through Week ${latest.week} · 2026`;
  }
}

function renderLasts(edge) {
  if (!edge) return;
  const o = edge.lasts.offense, d = edge.lasts.defense;
  mount('viz-lasts', {
    legend: [{ name: 'Offense', color: SERIES.model, dot: true }, { name: 'Defense', color: SERIES.vegas, dot: true }],
    table: { columns: ['Stat', 'Offense half to half', 'Defense half to half', 'Offense year to year', 'Defense year to year'], rows: o.map((x, i) => [x.stat, num(x.half), num(d[i].half), num(x.year), num(d[i].year)]) },
    draw: (host, w) => dotRows(host, w, {
      rows: o.map((x, i) => ({ label: x.stat, points: [{ name: 'Offense', value: x.half, color: SERIES.model }, { name: 'Defense', value: d[i].half, color: SERIES.vegas }] })),
      domain: [0, 0.65], ticks: [0, 0.2, 0.4, 0.6], fmt: v => num(v), connect: true, labelWidth: 140, rowHeight: 28, axisLabel: 'Correlation, first half of a season to the second',
    }),
  });
  const e = edge.explained;
  $('viz-explained-title').textContent = `About a third of a miss is turnovers, at ${e.points_per_turnover} points each`;
  const steps = [['turnovers', 'Turnovers'], ['plus_special_teams', '+ special teams'], ['plus_offense', '+ how well offenses moved the ball']];
  mount('viz-explained', {
    legend: [{ name: 'Our model', color: SERIES.model, dot: true }, { name: 'Vegas', color: SERIES.vegas, dot: true }],
    table: {
      columns: ['What happened in the game', 'Our model', 'Vegas'], rows: steps.map(([k, l]) => [l, pct(e.model[k], 0), pct(e.vegas[k], 0)]).concat(edge.misses.map(m => [`Next-miss slope: ${m.name}`, signed(m.slope, 3), `${signed(m.lo, 3)} to ${signed(m.hi, 3)}`])),
    },
    draw: (host, w) => dotRows(host, w, {
      rows: steps.map(([k, l]) => ({ label: l, points: [{ name: 'Our model', value: e.model[k], color: SERIES.model }, { name: 'Vegas', value: e.vegas[k], color: SERIES.vegas }] })),
      domain: [0, 1], ticks: [0, 0.25, 0.5, 0.75, 1], fmt: v => pct(v, 0), tickFmt: v => pct(v, 0), labelWidth: 190, rowHeight: 34, offset: 10, axisLabel: 'Share of the miss explained',
    }),
  });
}

function renderEdgeHunt(edge) {
  if (!edge) return;
  const r = edge.ratings, vars = r.variants.filter(v => v.diff !== undefined), best = vars.reduce((a, b) => (b.diff < a.diff ? b : a));
  $('viz-pbp-title').textContent = `${vars.length} richer ratings from play-by-play, no real gain`;
  $('viz-pbp-sub').textContent = `Change in squared error against v2 on ${r.games.toLocaleString()} games, each season held out. Left is better. The best, "${best.name}", gained ${Math.abs(best.diff).toFixed(1)} against a Vegas lead of about ${(r.variants[0].mse - r.vegas.mse).toFixed(0)}, so the rating is on a plateau. The table adds each version's weight beside Vegas: near zero means it adds nothing the line lacks.`;
  mount('viz-pbp', {
    table: { columns: ['Rating', 'Vs. v2', '95% range', 'Correlation', 'Avg miss', 'Winners', 'Vs. spread', 'Weight beside Vegas'], rows: r.variants.map(v => [v.name, v.diff === undefined ? '—' : signed(v.diff), v.diff === undefined ? '—' : `${signed(v.lo)} to ${signed(v.hi)}`, num(v.corr, 3), num(v.miss), pct(v.winners), pct(v.ats), `${signed(v.weight)} (${signed(v.weight_lo)} to ${signed(v.weight_hi)})`]) },
    draw: (host, w) => dotRows(host, w, {
      rows: vars.map(v => ({ label: v.name, points: [{ name: 'Vs. v2', value: v.diff, lo: v.lo, hi: v.hi, color: SERIES.model }] })),
      domain: [-4, 4], ticks: [-4, -2, 0, 2, 4], zero: 0, fmt: x => signed(x), labelWidth: 280, rowHeight: 26, axisLabel: 'Change in squared error (left is better)',
    }),
  });
  const rules = edge.rules;
  const all = [...rules.totals.map(x => ({ ...x, group: 'Totals: the under hits' })), ...rules.spread.map(x => ({ ...x, group: 'Spread: the side named covers' }))];
  const both = all.filter(x => x.old.rate > 0.524 && x.new.rate > 0.524);
  $('viz-rules-title').textContent = `${both.length} of ${all.length} simple rules clear break-even in both eras`;
  mount('viz-rules', {
    legend: [{ name: '2006–2018', color: SERIES.muted, dot: true }, { name: '2019–2026', color: SERIES.model, dot: true }],
    note: `Luck check, shuffling results within seasons 2,000 times: a rule as strong as "Wind 10+ mph" in both eras shows up in ${pct(rules.luck['Wind 10+ mph'], 0)} of runs. "Road underdogs" in ${pct(rules.luck['Road underdogs'], 0)}, and "Home underdogs of 7+" in ${pct(rules.luck['Home underdogs of 7+'], 0)}.`,
    table: { columns: ['Rule', '2006–18', 'Games', '2019–26', 'Games'], rows: all.map(x => [`${x.group.split(':')[0]}: ${x.name}`, pct(x.old.rate), x.old.n, pct(x.new.rate), x.new.n]) },
    draw: (host, w) => dotRows(host, w, {
      rows: all.map(x => ({ group: x.group, label: x.name, points: [{ name: '2006–2018', value: x.old.rate, color: SERIES.muted }, { name: '2019–2026', value: x.new.rate, color: SERIES.model }] })),
      domain: [0.35, 0.65], ticks: [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65], marker: 0.524, fmt: v => pct(v), tickFmt: v => pct(v, 0), connect: true, labelWidth: 190, rowHeight: 24, axisLabel: 'Hit rate. Line: break-even.',
    }),
  });
  mount('viz-moneyline', {
    legend: [{ name: '2006–2018', color: SERIES.muted, dot: true }, { name: '2019–2026', color: SERIES.model, dot: true }],
    table: { columns: ['Price', '2006–18', 'Games', '2019–26', 'Games'], rows: edge.moneyline.map(x => [x.name, signed(x.old.roi * 100, 1) + '%', x.old.n, signed(x.new.roi * 100, 1) + '%', x.new.n]) },
    draw: (host, w) => dotRows(host, w, {
      rows: edge.moneyline.map(x => ({ label: x.name, points: [{ name: '2006–2018', value: x.old.roi, color: SERIES.muted }, { name: '2019–2026', value: x.new.roi, color: SERIES.model }] })),
      domain: [-0.18, 0.06], ticks: [-0.15, -0.1, -0.05, 0, 0.05], zero: 0, fmt: v => `${signed(v * 100, 1)}%`, tickFmt: v => `${signed(v * 100, 0)}%`, connect: true, labelWidth: 170, rowHeight: 26, axisLabel: 'Return per dollar',
    }),
  });
  const c = edge.combined;
  $('viz-combined-title').textContent = `Every signal at once: ${pct(c.trained_on_the_past.rate)} against the spread`;
  $('viz-combined-sub').textContent = `A ridge model on all of the above, trained only on earlier seasons. That's right at break-even, and always taking the underdog hit ${pct(c.always_underdog.rate)} over the full span. Its most confident tenth did no better (${pct(c.top_decile.rate)}), so it can't power a "best bets" list.`;
  mount('viz-combined', {
    table: { columns: ['Season', 'Hit rate', '95% range', 'Games'], rows: c.by_season.map(x => [x.season, pct(x.rate), `${pct(x.lo)} to ${pct(x.hi)}`, x.n]) },
    draw: (host, w) => dotRows(host, w, {
      rows: c.by_season.map(x => ({ label: String(x.season), points: [{ name: 'Hit rate', value: x.rate, lo: x.lo, hi: x.hi, color: SERIES.model }] })),
      domain: [0.35, 0.7], ticks: [0.4, 0.5, 0.6, 0.7], marker: 0.524, fmt: v => pct(v), tickFmt: v => pct(v, 0), labelWidth: 56, rowHeight: 30, axisLabel: 'Spread picks right. Line: break-even.',
    }),
  });
}

function renderInventory(edge, lab, wind) {
  if (!edge) return;
  const i = edge.inventory, rows = [
    ['nflverse play-by-play', `${i.plays.toLocaleString()} pass and rush plays, ${i.team_games.toLocaleString()} team-games`, `${i.pbp_seasons[0]}–${i.pbp_seasons[1]}`, 'Team ratings, verdicts, what lasts'],
    ['nflverse schedules and closing lines', `${i.market_games.toLocaleString()} games with a line`, `${i.market_seasons[0]}–${i.market_seasons[1]}`, 'Factor lab, market rules, every grade'],
    ['nflverse weekly player stats, rosters, injuries', lab ? `${lab.weekly.ppr.position_weeks.toLocaleString()} position-weeks ranked` : 'Weekly', '2018–2026', 'Start Score, draft board'],
    ['FantasyPros consensus rankings', 'Draft and weekly, every position', 'Late 2020 on', 'The expert baseline, streaming test'],
    ['Open-Meteo forecasts', wind && wind.backtest ? `${wind.backtest.forecast.games} archived games, plus live hourly wind` : 'Archived and live', '2022 on', 'Wind Watch'],
    ['National Weather Service', 'Point forecasts for each outdoor stadium', 'Live', 'Game card weather'],
  ];
  const t = html('table', 'data-table inventory-table', undefined, $('inventory'));
  const head = html('tr', '', undefined, html('thead', '', undefined, t));
  ['Source', 'What', 'Coverage', 'Feeds'].forEach(c => html('th', 'left', c, head));
  const body = html('tbody', '', undefined, t);
  for (const row of rows) {
    const tr = html('tr', '', undefined, body);
    row.forEach(v => html('td', 'wrap', v, tr));
  }
}

async function main() {
  const load = url => fetch(url).then(r => (r.ok ? r.json() : null)).catch(() => null);
  const [acc, factors, rating, compare, lab, weeklyAcc, wind, profiles, stream, rankings, inputs, board, edge] = await Promise.all([
    load('data/betting/accountability.json'), load('data/lab/factor_lab.json'), load('data/lab/rating_lab.json'),
    load('data/lab/model_compare.json'), load('data/lab/fantasy_lab.json'), load('data/fantasy/weekly_accountability.json'),
    load('data/betting/wind_watch.json'), load('data/betting/team_profiles.json'), load('data/lab/streaming_lab.json'),
    load('data/fantasy/weekly_rankings.json'), load('data/fantasy/start_score_inputs.json'), load('data/betting/game_board.json'), load('data/lab/edge_lab.json'),
  ]);
  const week = wind && wind.week ? await load(`data/betting/archive/wind/${wind.season}-w${String(wind.week).padStart(2, '0')}.json`) : null;
  // One section's missing file shouldn't take down the rest of the page.
  const sections = [
    () => renderFound(wind, profiles, stream, lab), () => renderWind(wind, week, edge), () => renderTeams(profiles), () => renderStreaming(rankings, stream),
    () => renderScoreExplorer(inputs), () => renderFantasy(lab, weeklyAcc), () => renderGameModel(board, compare, acc), () => renderSeason(acc, compare),
    () => renderRating(rating), () => renderMotivation(factors), () => renderLasts(edge), () => renderEdgeHunt(edge), () => renderInventory(edge, lab, wind),
  ];
  for (const run of sections) {
    try { run(); } catch (err) { console.error(err); }
  }
  if (lab) {
    const when = iso => new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    $('lab-dates').textContent = `The test scripts last ran ${when(lab.generated_at)}; the live charts refresh with every scan.`;
  }
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
}

main();
