const $ = (s, r = document) => r.querySelector(s);
const el = (t, a = {}, ...k) => { const e = document.createElement(t); for (const [x, y] of Object.entries(a)) x === 'class' ? e.className = y : x.startsWith('on') ? e.addEventListener(x.slice(2), y) : e.setAttribute(x, y); k.flat().forEach(c => e.append(c?.nodeType ? c : document.createTextNode(c ?? ''))); return e; };

const PRESETS = [
  { n: 'Clear day', s: 'baseline', seed: 12, flood: 0, cloud: 0, drought: 0 },
  { n: 'Flash flood', s: 'clear sky', seed: 7, flood: 0.8, cloud: 0.05, drought: 0 },
  { n: 'Monsoon flood under cloud', s: 'optical blinded', seed: 7, flood: 0.8, cloud: 0.7, drought: 0.1 },
  { n: 'Drought in haze', s: 'crops stressed', seed: 31, flood: 0, cloud: 0.3, drought: 0.9 },
  { n: 'River delta', s: 'mixed hazards', seed: 52, flood: 0.55, cloud: 0.4, drought: 0.3 },
];
const SENS = [
  { k: 'optical', n: 'Optical', d: '6 bands · Sentinel-2 like', layer: 'rgb' },
  { k: 'sar', n: 'Radar (SAR)', d: 'VV + VH · sees through cloud', layer: 'sar' },
  { k: 'dem', n: 'Terrain (DEM)', d: 'elevation + slope', layer: 'dem' },
];
const LAYERS = [
  { g: 'Inputs', items: [
    { k: 'rgb', n: 'True colour', sensor: 0 }, { k: 'fcc', n: 'False colour', sensor: 0 }, { k: 'ndvi', n: 'NDVI', sensor: 0 },
    { k: 'sar', n: 'Radar', sensor: 1 }, { k: 'dem', n: 'Terrain', sensor: 2 }] },
  { g: 'Model output', items: [
    { k: 'lc', n: 'Land cover', pair: 'lc_true' }, { k: 'flood', n: 'Flood', pair: 'flood_true' },
    { k: 'veg', n: 'Vegetation', pair: 'veg_true' }, { k: 'unc', n: 'Uncertainty' }] },
  { g: 'Foundation features', items: [
    { k: 'recon', n: 'Cloud-free', pair: 'rgb', pairLabel: ['Reconstructed', 'Cloudy input'], always: true },
    { k: 'pca', n: 'Embeddings', pixelated: true }, { k: 'sim', n: 'Similarity' }] },
];
const SUBS = {
  rgb: 'Bands B4-B3-B2 as the satellite saw it', fcc: 'NIR-red-green: vegetation glows red', ndvi: 'Greenness computed from the optical input',
  sar: 'R=VV, G=VH, B=VV−VH — water is black, buildings are bright', dem: 'Elevation tint with hillshade',
  lc: 'Six classes predicted per pixel', flood: 'Predicted floodwater (cyan) over the reconstructed scene', veg: 'Predicted vegetation health index',
  unc: 'Normalised entropy of the land-cover prediction', recon: 'Clear pixels kept; clouded pixels rebuilt from SAR, terrain and context',
  pca: 'Top 3 principal components of the encoder tokens as RGB', sim: 'Click anywhere: cosine similarity of that spot to every other',
};
const LEGEND = {
  flood: [['#14DCFF', 'Floodwater'], ['#7850FF', 'Flood probability']],
  veg: { bar: 'linear-gradient(90deg,#a50026,#f46d43,#fee08b,#a6d96a,#006837)', a: 'Stressed', b: 'Healthy' },
  ndvi: { bar: 'linear-gradient(90deg,#a50026,#f46d43,#fee08b,#a6d96a,#006837)', a: '−0.2', b: '0.9' },
  unc: { bar: 'linear-gradient(90deg,#000004,#51127c,#b73779,#fc8761,#fcfdbf)', a: 'Confident', b: 'Unsure' },
  dem: { bar: 'linear-gradient(90deg,#333399,#00b3b3,#7ed97e,#f0e68c,#a0764f,#fff)', a: 'Low', b: 'High' },
  pca: [['#888', 'Similar colour = similar land surface']],
  sim: [['#fcffa4', 'Similar'], ['#420a68', 'Different']],
};

const S = { seed: 7, flood: .55, cloud: .4, drought: .3, sensors: [true, true, true], source: 'sim', scene_id: null,
  layer: 'recon', compare: true, res: null, pix: {}, cmpPos: .5, sim: [64, 64], req: 0 };

// ---------------- navigation
document.querySelectorAll('.tabs button').forEach(b => b.onclick = () => {
  document.querySelectorAll('.tabs button').forEach(x => x.setAttribute('aria-selected', x === b));
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === 'page-' + b.dataset.page));
  history.replaceState(null, '', '#' + b.dataset.page);
  if (b.dataset.page === 'bench') loadBench();
});

// ---------------- controls
const presetsUl = $('#presets');
PRESETS.forEach((p, i) => presetsUl.append(el('li', {}, el('button', { 'data-i': i, onclick: () => applyPreset(i) }, p.n, el('small', {}, p.s)))));
function applyPreset(i) {
  const p = PRESETS[i]; Object.assign(S, { seed: p.seed, flood: p.flood, cloud: p.cloud, drought: p.drought, source: 'sim' });
  syncControls(); run();
}
function syncControls() {
  $('#seed').value = S.seed;
  for (const k of ['flood', 'cloud', 'drought']) { $('#' + k).value = S[k]; $('#o-' + k).textContent = (+S[k]).toFixed(2); }
  presetsUl.querySelectorAll('button').forEach((b, i) => { const p = PRESETS[i]; b.classList.toggle('on', S.source === 'sim' && p.seed === +S.seed && p.flood === +S.flood && p.cloud === +S.cloud && p.drought === +S.drought); });
  document.querySelectorAll('.sensor').forEach((b, i) => b.setAttribute('aria-pressed', S.sensors[i]));
}
let debounce;
const later = () => { clearTimeout(debounce); debounce = setTimeout(run, 220); };
for (const k of ['flood', 'cloud', 'drought']) $('#' + k).oninput = e => { S[k] = +e.target.value; S.source = 'sim'; $('#o-' + k).textContent = S[k].toFixed(2); syncControls(); later(); };
$('#seed').onchange = e => { S.seed = Math.max(0, +e.target.value | 0); S.source = 'sim'; syncControls(); run(); };
$('#shuffle').onclick = () => { S.seed = Math.floor(Math.random() * 99999); S.source = 'sim'; syncControls(); run(); };
$('#file').onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/api/upload', { method: 'POST', body: fd });
  if (!r.ok) return toast((await r.json()).detail || 'Upload failed');
  S.scene_id = (await r.json()).scene_id; S.source = 'upload'; syncControls(); run(); e.target.value = '';
};
function toast(m) { $('#readout').textContent = m; $('#readout').style.color = '#E8452B'; setTimeout(() => $('#readout').style.color = '', 4000); }

const sensorBox = $('#sensors');
SENS.forEach((s, i) => sensorBox.append(el('button', { class: 'sensor', 'aria-pressed': 'true', onclick: () => {
  if (S.sensors[i] && S.sensors.filter(Boolean).length === 1) return toast('Keep at least one sensor on.');
  S.sensors[i] = !S.sensors[i]; syncControls(); later();
} }, el('img', { alt: '', id: 'sth-' + s.k }), el('div', {}, el('b', {}, s.n), el('span', {}, s.d)), el('i', { class: 'sw' }))));

// ---------------- run analysis
async function run() {
  const id = ++S.req; $('#busy').classList.add('on');
  const body = { source: S.source, seed: +S.seed, size: 128, flood: S.flood, cloud: S.cloud, drought: S.drought, sensors: S.sensors, scene_id: S.scene_id };
  let r;
  try { r = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }); }
  catch { $('#busy').classList.remove('on'); return toast('Server unreachable.'); }
  if (id !== S.req) return;
  if (!r.ok) { $('#busy').classList.remove('on'); return toast((await r.json()).detail || 'Analysis failed'); }
  const d = await r.json();
  S.res = d; S.pix = {}; S.size = d.stats.size;
  for (const [k, v] of Object.entries(d.pix)) if (v) S.pix[k] = Uint8Array.from(atob(v), c => c.charCodeAt(0));
  S.truth = d.truth; if (!d.layers.includes(S.layer) && S.layer !== 'sim') S.layer = 'rgb';
  S.sim = [Math.floor(S.size[1] / 2), Math.floor(S.size[0] / 2)];
  renderStrip(); await showLayer(); renderFindings(); $('#busy').classList.remove('on');
  SENS.forEach(s => $('#sth-' + s.k).src = layerUrl(s.layer));
  $('#export').href = `/api/export/${d.id}.npz`;
}
const layerUrl = k => k === 'sim' ? `/api/similarity/${S.res.id}.png?x=${S.sim[0]}&y=${S.sim[1]}` : `/api/layer/${S.res.id}/${k}.png`;
const findLayer = k => LAYERS.flatMap(g => g.items).find(i => i.k === k);

function renderStrip() {
  const strip = $('#strip'); strip.innerHTML = '';
  LAYERS.forEach(g => strip.append(el('div', { class: 'sgrp' }, el('b', {}, g.g), el('div', {}, g.items.map(it =>
    el('button', { class: 'chip', role: 'tab', 'aria-selected': it.k === S.layer, 'data-k': it.k, onclick: () => { S.layer = it.k; S.compare = !!(it.always); document.querySelectorAll('.chip').forEach(c => c.setAttribute('aria-selected', c.dataset.k === it.k)); showLayer(); } },
      el('img', { alt: '', src: layerUrl(it.k), loading: 'lazy' }), el('span', {}, it.n)))))));
}
const preload = u => new Promise(res => { const i = new Image(); i.onload = i.onerror = () => res(); i.src = u; });
async function showLayer() {
  const it = findLayer(S.layer), b = $('#img-base'), o = $('#img-over');
  const pairKey = it.pair && (S.truth || it.always) ? it.pair : null;
  const useCmp = !!pairKey && S.compare;
  const u = layerUrl(S.layer), u2 = pairKey ? layerUrl(pairKey) : null;
  await Promise.all([preload(u), u2 ? preload(u2) : 0]);
  b.src = u; b.classList.toggle('px', !!it.pixelated || S.layer === 'lc');
  o.style.display = useCmp ? 'block' : 'none'; if (useCmp) { o.src = u2; o.classList.toggle('px', S.layer === 'lc'); }
  const cmpBtn = $('#cmp-btn'); cmpBtn.style.display = pairKey ? '' : 'none'; cmpBtn.setAttribute('aria-pressed', useCmp);
  cmpBtn.textContent = it.always ? 'Compare with input' : 'Compare with truth';
  $('#divider').hidden = !useCmp; $('#tag-l').hidden = $('#tag-r').hidden = !useCmp;
  if (useCmp) { const lab = it.pairLabel || ['Prediction', 'Ground truth']; $('#tag-l').textContent = lab[0]; $('#tag-r').textContent = lab[1]; setCmp(S.cmpPos); }
  $('#layer-title').textContent = it.n; $('#layer-sub').textContent = SUBS[S.layer] || '';
  const off = it.sensor !== undefined && !S.sensors[it.sensor]; $('#offline').hidden = !off;
  $('#cross').hidden = S.layer !== 'sim'; if (S.layer === 'sim') placeCross();
  renderLegend();
}
$('#cmp-btn').onclick = () => { S.compare = !S.compare; showLayer(); };

function setCmp(p) { S.cmpPos = Math.max(0, Math.min(1, p)); $('#img-over').style.clipPath = `inset(0 0 0 ${S.cmpPos * 100}%)`; $('#divider').style.left = S.cmpPos * 100 + '%'; }
let drag = false;
const rel = e => { const r = $('#viewer').getBoundingClientRect(); return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]; };
$('#viewer').addEventListener('pointerdown', e => {
  const [x, y] = rel(e);
  if (!$('#divider').hidden && (e.target.closest('#divider') || true)) { drag = true; $('#viewer').setPointerCapture(e.pointerId); setCmp(x); return; }
  if (S.layer === 'sim') pickSim(x, y);
});
$('#viewer').addEventListener('pointermove', e => { const [x, y] = rel(e); if (drag) setCmp(x); readout(x, y); });
$('#viewer').addEventListener('pointerup', () => drag = false);
$('#viewer').addEventListener('click', e => { if (S.layer === 'sim' && $('#divider').hidden) { const [x, y] = rel(e); pickSim(x, y); } });
function pickSim(x, y) { S.sim = [Math.min(S.size[1] - 1, Math.floor(x * S.size[1])), Math.min(S.size[0] - 1, Math.floor(y * S.size[0]))]; placeCross(); $('#img-base').src = layerUrl('sim'); }
function placeCross() { const c = $('#cross'); c.style.left = (S.sim[0] + .5) / S.size[1] * 100 + '%'; c.style.top = (S.sim[1] + .5) / S.size[0] * 100 + '%'; }

function readout(x, y) {
  if (!S.res) return;
  const [H, W] = S.size, px = Math.min(W - 1, Math.max(0, Math.floor(x * W))), py = Math.min(H - 1, Math.max(0, Math.floor(y * H))), i = py * W + px;
  const cls = S.res.classes, p = S.pix;
  let t = `x ${px}, y ${py}  ·  ${cls[p.lc[i]].name}`;
  if (p.lc_true) t += ` (truth ${cls[p.lc_true[i]].name})`;
  t += `  ·  flood ${(p.flood[i] / 2.55).toFixed(0)}%  ·  vegetation ${(p.veg[i] / 255).toFixed(2)}  ·  uncertainty ${(p.unc[i] / 2.55).toFixed(0)}%`;
  $('#readout').textContent = t;
}
function renderLegend() {
  const lg = $('#legend'); lg.innerHTML = ''; const k = S.layer;
  if (k === 'lc') S.res.classes.forEach(c => lg.append(el('span', {}, el('i', { style: `background:${c.color}` }), c.name)));
  else if (LEGEND[k]?.bar) lg.append(el('span', {}, LEGEND[k].a, el('i', { class: 'bar', style: `background:${LEGEND[k].bar}` }), LEGEND[k].b));
  else if (LEGEND[k]) LEGEND[k].forEach(([c, t]) => lg.append(el('span', {}, el('i', { style: `background:${c}` }), t)));
}

// ---------------- findings
const pct = v => (v * 100).toFixed(1) + '%';
function meter(label, v, color, main) {
  return el('div', { class: 'meter' + (main ? ' main' : '') }, el('label', {}, label, el('b', {}, Math.round(v) + ' / 100')),
    el('div', { class: 'track' }, el('div', { class: 'fill', style: `width:${v}%;background:${color}` })));
}
const riskColor = v => v < 25 ? '#3DAE7A' : v < 50 ? '#F2B14C' : v < 75 ? '#E8742B' : '#E8452B';
function renderFindings() {
  const st = S.res.stats, r = st.risk;
  $('#risk').replaceChildren(meter('Composite risk', r.composite, riskColor(r.composite), true), meter('Flood exposure', r.flood, '#1FB6FF'), meter('Vegetation stress', r.drought, '#D9822B'), meter('Urban pressure', r.urban, '#7A8794'));
  const c = $('#comp'); c.innerHTML = '';
  c.append(el('div', { class: 'stack' }, st.composition.map((x, i) => el('div', { style: `flex:${Math.max(x.frac, .002)};background:${S.res.classes[i].color}`, title: x.name }))));
  c.append(el('table', {}, st.composition.slice().sort((a, b) => b.frac - a.frac).map(x => {
    const i = st.composition.indexOf(x);
    return el('tr', {}, el('td', {}, el('i', { style: `background:${S.res.classes[i].color}` }), x.name), el('td', { class: 'p' }, pct(x.frac)), el('td', {}, x.km2.toFixed(2) + ' km²'));
  })));
  $('#insights').replaceChildren(...st.insights.map(i => el('li', { class: i.kind }, i.text)));
  const a = st.accuracy, t = $('#acc'); t.innerHTML = '';
  $('#acc-h').style.display = t.style.display = a ? '' : 'none';
  if (a) [['Land-cover mIoU', a.miou.toFixed(3)], ['Pixel accuracy', pct(a.oa)], ['Flood IoU', a.flood_iou == null ? 'no flood' : a.flood_iou.toFixed(3)], ['Vegetation MAE', a.veg_mae.toFixed(3)], ['Cloud cover (true)', pct(a.true_cloud ?? 0)]]
    .forEach(([k, v]) => t.append(el('tr', {}, el('td', {}, k), el('td', {}, v))));
  $('#timing').textContent = `${st.size[1]}×${st.size[0]} px at ${st.gsd_m} m · ${st.area_km2} km² · inference ${st.timing_ms} ms on CPU`;
}

// ---------------- status
async function health() {
  const s = $('#status'); try { const h = await (await fetch('/api/health')).json();
    s.className = 'status ' + (h.ready ? 'ok' : 'bad'); s.lastChild.textContent = h.ready ? `Model online · ${(h.params / 1e6).toFixed(2)}M parameters` : 'Model not trained yet'; return h.ready; }
  catch { s.className = 'status bad'; s.lastChild.textContent = 'Server offline'; return false; }
}

// ---------------- benchmarks
let benchLoaded = false;
function svgEl(t, a = {}, txt) { const e = document.createElementNS('http://www.w3.org/2000/svg', t); for (const [k, v] of Object.entries(a)) e.setAttribute(k, v); if (txt != null) e.textContent = txt; return e; }
function lineChart(series, { xl, yl, ymin, ymax, fmt = v => v.toFixed(2), W = 520, H = 280 }) {
  const m = { l: 44, r: 12, t: 10, b: 34 }, svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  const xs = series.flatMap(s => s.pts.map(p => p[0])), ys = series.flatMap(s => s.pts.map(p => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = ymin ?? Math.min(...ys), y1 = ymax ?? Math.max(...ys);
  const X = v => m.l + (v - x0) / (x1 - x0 || 1) * (W - m.l - m.r), Y = v => H - m.b - (v - y0) / (y1 - y0 || 1) * (H - m.t - m.b);
  for (let i = 0; i <= 4; i++) { const v = y0 + (y1 - y0) * i / 4; svg.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: '#CBD5D8' }), svgEl('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, fmt(v))); }
  for (let i = 0; i <= 4; i++) { const v = x0 + (x1 - x0) * i / 4; svg.append(svgEl('text', { x: X(v), y: H - m.b + 16, 'text-anchor': 'middle' }, Math.round(v))); }
  svg.append(svgEl('text', { x: (W + m.l) / 2, y: H - 4, 'text-anchor': 'middle' }, xl));
  series.forEach(s => { svg.append(svgEl('path', { d: s.pts.map((p, i) => (i ? 'L' : 'M') + X(p[0]).toFixed(1) + ' ' + Y(p[1]).toFixed(1)).join(''), fill: 'none', stroke: s.color, 'stroke-width': 2.5, 'stroke-linejoin': 'round' }));
    const l = s.pts[s.pts.length - 1]; svg.append(svgEl('circle', { cx: X(l[0]), cy: Y(l[1]), r: 4, fill: s.color })); });
  return svg;
}
function barChart(groups, series, { W = 520, H = 290, fmt = v => v.toFixed(2) }) {
  const m = { l: 36, r: 8, t: 14, b: 44 }, svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' }), gw = (W - m.l - m.r) / groups.length;
  const Y = v => H - m.b - v * (H - m.t - m.b);
  for (let i = 0; i <= 4; i++) { const v = i / 4; svg.append(svgEl('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: '#CBD5D8' }), svgEl('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end' }, v.toFixed(2))); }
  groups.forEach((g, i) => { const bw = gw * .72 / series.length;
    series.forEach((s, j) => { const v = s.vals[i], x = m.l + i * gw + gw * .14 + j * bw;
      svg.append(svgEl('rect', { x, y: Y(v), width: bw - 2, height: Y(0) - Y(v), fill: s.color, rx: 2 }), svgEl('text', { x: x + bw / 2 - 1, y: Y(v) - 4, 'text-anchor': 'middle' }, fmt(v).replace(/^0/, ''))); });
    svg.append(svgEl('text', { x: m.l + i * gw + gw / 2, y: H - m.b + 16, 'text-anchor': 'middle' }, g.split(' ')[0]), svgEl('text', { x: m.l + i * gw + gw / 2, y: H - m.b + 29, 'text-anchor': 'middle' }, g.split(' ').slice(1).join(' '))); });
  return svg;
}
const chartBox = (t, p, svg, legend) => el('div', { class: 'chart' }, el('h3', {}, t), el('p', {}, p), svg, legend ? el('div', { class: 'lg' }, legend.map(([c, n]) => el('span', {}, el('i', { style: `background:${c}` }), n))) : '');
async function loadBench() {
  if (benchLoaded) return; const body = $('#bench-body');
  const r = await fetch('/api/benchmarks'); if (!r.ok) { body.textContent = 'Benchmarks not available yet — run the training pipeline.'; return; }
  const B = await r.json(); benchLoaded = true; body.innerHTML = '';
  const T = '#0F6E7A', W = '#D9822B', f = B.ablation['All sensors'], s = B.scratch['All sensors'];
  $('#b-title').textContent = `Pretraining lifts land-cover mIoU from ${s.miou.toFixed(2)} to ${f.miou.toFixed(2)} with ${B.config.labels} labelled scenes.`;
  const hc = B.heavy_cloud, cm = f.cloud_rmse;
  body.append(el('div', { class: 'hero-nums' },
    el('div', { class: 'hn' }, el('div', { class: 'v' }, f.miou.toFixed(3), el('small', {}, `vs ${s.miou.toFixed(3)} scratch`)), el('p', {}, 'Land-cover mean IoU, all sensors')),
    el('div', { class: 'hn' }, el('div', { class: 'v' }, f.flood_iou.toFixed(3), el('small', {}, `vs ${s.flood_iou.toFixed(3)}`)), el('p', {}, 'Flood extent IoU')),
    el('div', { class: 'hn' }, el('div', { class: 'v' }, f.veg_mae.toFixed(3), el('small', {}, `vs ${s.veg_mae.toFixed(3)}`)), el('p', {}, 'Vegetation-health error (MAE, lower is better)')),
    el('div', { class: 'hn' }, el('div', { class: 'v' }, cm ? (cm * 100).toFixed(1) + '%' : '–', el('small', {}, cm ? `vs ${(f.cloud_rmse_input * 100).toFixed(1)}% cloudy` : '')), el('p', {}, 'Reflectance error under clouds after reconstruction'))));
  const pre = B.curves.pretrain.map(d => [d.step, d.loss]);
  const va = n => B.curves[n].filter(d => d.val_miou != null).map(d => [d.step, d.val_miou]);
  const vf = n => B.curves[n].filter(d => d.val_flood_iou != null).map(d => [d.step, d.val_flood_iou]);
  const sets = Object.keys(B.ablation);
  body.append(el('div', { class: 'chartgrid' },
    chartBox('Self-supervised pretraining', 'Masked, cloud-contaminated, sensor-dropped scenes in; every channel and the cloud-free image out. No labels used.', lineChart([{ pts: pre, color: T }], { xl: 'training step', ymin: 0 }), null),
    chartBox('Fine-tuning curves', `Validation land-cover mIoU while training on ${B.config.labels} labelled scenes.`, lineChart([{ pts: va('scratch'), color: W }, { pts: va('finetuned'), color: T }], { xl: 'fine-tuning step', ymin: 0, ymax: 1 }), [[T, 'Pretrained TerraFM'], [W, 'Same network from scratch']]),
    chartBox('Fine-tuning curves: flood', 'Validation flood IoU.', lineChart([{ pts: vf('scratch'), color: W }, { pts: vf('finetuned'), color: T }], { xl: 'fine-tuning step', ymin: 0, ymax: 1 }), [[T, 'Pretrained TerraFM'], [W, 'From scratch']]),
    chartBox('One model, any sensor subset', 'Land-cover mIoU when only some sensors are available at test time.', barChart(sets, [{ vals: sets.map(k => B.scratch[k].miou), color: W }, { vals: sets.map(k => B.ablation[k].miou), color: T }], {}), [[T, 'Pretrained TerraFM'], [W, 'From scratch']])));
  const best = (k, lower) => v => v;
  const rows = sets.map(k => { const a = B.ablation[k], b = B.scratch[k];
    return el('tr', {}, el('td', {}, k), [[a.miou, b.miou, 1], [a.oa, b.oa, 1], [a.flood_iou, b.flood_iou, 1], [a.veg_mae, b.veg_mae, 0]].map(([x, y, hi]) =>
      el('td', {}, el('span', { class: (hi ? x >= y : x <= y) ? 'best' : '' }, x.toFixed(3)), el('span', { style: 'color:var(--mute)' }, ' / ' + y.toFixed(3))))); });
  body.append(el('div', {}, el('h3', { style: 'font-size:18px;margin-bottom:8px' }, 'Every sensor combination'), el('p', { style: 'color:var(--mute);font-size:13.5px;margin:0 0 8px' }, 'Each cell: TerraFM / scratch. Teal marks the better value.'),
    el('div', { class: 'tw' }, el('table', { class: 't' }, el('thead', {}, el('tr', {}, ['Sensors', 'mIoU', 'Accuracy', 'Flood IoU', 'Veg MAE'].map(h => el('th', {}, h)))), el('tbody', {}, rows)))));
  const hr = (n, m) => el('tr', {}, el('td', {}, n), el('td', {}, m.miou.toFixed(3)), el('td', {}, m.flood_iou.toFixed(3)), el('td', {}, m.veg_mae.toFixed(3)));
  body.append(el('div', {}, el('h3', { style: 'font-size:18px;margin-bottom:8px' }, `Heavily clouded scenes (${hc.n} scenes, over 30% cover)`), el('p', { style: 'color:var(--mute);font-size:13.5px;margin:0 0 8px' }, 'Radar and terrain keep the model working when optical is unreliable.'),
    el('div', { class: 'tw' }, el('table', { class: 't' }, el('thead', {}, el('tr', {}, ['Model · input', 'mIoU', 'Flood IoU', 'Veg MAE'].map(h => el('th', {}, h)))),
      el('tbody', {}, hr('TerraFM · optical only', hc.finetuned_optical), hr('TerraFM · all sensors', hc.finetuned_all), hr('Scratch · optical only', hc.scratch_optical), hr('Scratch · all sensors', hc.scratch_all))))));
  const z = B.size128.finetuned;
  body.append(el('p', { style: 'color:var(--ink2);max-width:70ch' }, `Trained on 64×64 crops, TerraFM runs unchanged on 128×128 scenes (fixed sin-cos positions): mIoU ${z.miou.toFixed(3)}, flood IoU ${z.flood_iou.toFixed(3)}.`));
}

// ---------------- model page
function buildModelPage() {
  const b = $('#model-body');
  b.innerHTML = `
  <div><h2>Architecture</h2>
  <svg class="flow" viewBox="0 0 1040 330" role="img" aria-label="Architecture diagram">
    <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#41505F"/></marker></defs>
    ${[['Optical','6 bands',40,'#F2B14C'],['Radar','VV, VH',130,'#21C6D6'],['Terrain','elev, slope',220,'#7DAF6B']].map(([n,s,y,c])=>`<rect x="10" y="${y}" width="130" height="64" rx="6" fill="${c}" opacity=".22" stroke="${c}"/><text x="26" y="${y+27}">${n}</text><text class="s" x="26" y="${y+47}">${s}</text><path d="M140 ${y+32}H190" stroke="#41505F" marker-end="url(#ar)"/>`).join('')}
    <rect x="190" y="30" width="150" height="260" rx="6" fill="#fff" stroke="#CBD5D8"/><text x="206" y="58">Patch embedders</text><text class="s" x="206" y="78">one per sensor, 4×4</text><text class="s" x="206" y="110">absent sensor →</text><text class="s" x="206" y="126">left out of the mean</text><text class="s" x="206" y="158">+ sensor tags</text><text class="s" x="206" y="174">+ 2-D sin-cos position</text>
    <path d="M340 160H380" stroke="#41505F" marker-end="url(#ar)"/>
    <rect x="380" y="30" width="190" height="260" rx="6" fill="#0E1B2B"/><text x="398" y="60" style="fill:#fff;font-weight:700">Transformer encoder</text><text class="s" x="398" y="82" style="fill:#9FB2BC">4 blocks · 96 dim · 4 heads</text><text class="s" x="398" y="104" style="fill:#9FB2BC">shared across all sensors</text>
    ${[0,1,2,3].map(i=>`<rect x="398" y="${130+i*34}" width="154" height="24" rx="4" fill="#1B3045" stroke="#21C6D6" stroke-opacity=".5"/>`).join('')}
    <path d="M570 110H620" stroke="#41505F" marker-end="url(#ar)"/><path d="M570 220H620" stroke="#41505F" marker-end="url(#ar)"/>
    <rect x="620" y="40" width="190" height="110" rx="6" fill="#fff" stroke="#CBD5D8"/><text x="636" y="68">Cross-modal decoder</text><text class="s" x="636" y="90">rebuilds all 10 channels</text><text class="s" x="636" y="108">incl. cloud-free optical</text><text class="s" x="636" y="126">(pretraining objective)</text>
    <rect x="620" y="170" width="190" height="120" rx="6" fill="#fff" stroke="#CBD5D8"/><text x="636" y="198">Task decoder</text><text class="s" x="636" y="220">upsample ×4 + pixel skip</text><text class="s" x="636" y="238">fine-tuned on few labels</text>
    <path d="M810 95H850" stroke="#41505F" marker-end="url(#ar)"/><path d="M810 230H850" stroke="#41505F" marker-end="url(#ar)"/>
    <text x="860" y="90">Cloud-free image</text><text class="s" x="860" y="108">Embedding maps</text>
    <text x="860" y="196">Land cover (6)</text><text x="860" y="218">Flood extent</text><text x="860" y="240">Vegetation health</text>
  </svg></div>
  <div class="twocol"><div><h2>Pretraining without labels</h2><ul class="steps">
    <li><b>Mask.</b> 40–80% of image patches are hidden from the encoder.</li>
    <li><b>Drop sensors.</b> Whole sensors disappear at random, so the model learns to predict one from the others.</li>
    <li><b>Add clouds.</b> Optical inputs carry clouds and shadows, but the target is the clean image. The model learns to see through them using radar and terrain.</li></ul></div>
    <div><h2>Fine-tuning with few labels</h2><ul class="steps">
    <li><b>Three heads, one encoder.</b> Land cover, flood extent and vegetation health train together.</li>
    <li><b>Sensor dropout continues,</b> so every sensor subset works at inference.</li>
    <li><b>Reconstruction stays on</b> as an auxiliary loss, so the cloud-free output remains in the final model.</li></ul></div></div>
  <div><h2>Using your own data</h2><p>Upload an <code>.npz</code> with <code>optical</code> (6×H×W reflectance 0–1: B2, B3, B4, B8, B11, B12), <code>sar</code> (2×H×W, VV and VH in dB) and <code>dem</code> (2×H×W: elevation in metres, slope in degrees). To train on real tiles, replace <code>terrascope/core/simulator.py</code> with a loader that returns the same arrays and labels, then run <code>./run_training.sh</code>.</p>
  <p>The simulator mimics the physics that matters here: spectral signatures and vegetation vigour, speckled radar backscatter, double-bounce over flooded buildings, terrain shading, and cloud and shadow contamination. Scores are measured on unseen simulated scenes, not on real imagery.</p></div>`;
}

// ---------------- boot
(async () => {
  buildModelPage(); syncControls();
  const ok = await health();
  const h = location.hash.slice(1); if (['bench', 'model'].includes(h)) document.querySelector(`.tabs [data-page=${h}]`).click();
  if (ok) { await run(); }
  else $('#readout').textContent = 'Train the model first: ./run_training.sh';
})();
