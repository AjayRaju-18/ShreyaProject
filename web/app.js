// Fake Review Checker — runs the CNN, BERT and XLNet models entirely in the browser.
// Model files live in ./models/ (uploaded by notebooks/06_deploy_web.ipynb).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import { BertTokenizer, PreTrainedTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';
import { cnnProb, cnnExplain, unpackWeights } from './cnn.js';
import { createVader } from './vader.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
// Tokenizers are built straight from tokenizer.json (AutoTokenizer's local-path lookup fails in
// transformers.js 4.3); XLNet has no dedicated class there, and the generic one reads every rule
// from tokenizer.json. Both were checked to give the same token ids as the Python tokenizers.
const TOKENIZER_CLASS = { bert: BertTokenizer, xlnet: PreTrainedTokenizer };

// The website labels a review FAKE when the fake probability is 50% or more. (The notebooks use stricter
// validation-tuned thresholds, e.g. 0.90 for XLNet, which almost never fire on typed reviews.)
const CUTOFF = 0.5;
// Probability of the predicted label (fake prob for FAKE, 1 - fake prob for GENUINE), and its display text.
const labelProb = (p) => (p >= CUTOFF ? p : 1 - p);
const shownPct = (p) => `${(labelProb(p) * 100).toFixed(1)}% ${p >= CUTOFF ? 'fake' : 'genuine'}`;
const MODELS = {};          // name -> { prob: async (text, title, rating) => number, threshold }
const $ = (id) => document.getElementById(id);
let sentiment;              // VADER compound score, set up in loadCNN()

// ---------- download helpers with progress ----------
const statusRows = {};
function setStatus(name, msg) {
  if (!statusRows[name]) {
    const row = document.createElement('div');
    row.innerHTML = `<span>${name}</span><span></span>`;
    $('status').appendChild(row);
    statusRows[name] = row.lastChild;
  }
  statusRows[name].textContent = msg;
}

async function fetchWithProgress(url, name) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    setStatus(name, total ? `downloading ${Math.round(got / total * 100)}% of ${(total / 1e6).toFixed(0)} MB`
                          : `downloading ${(got / 1e6).toFixed(0)} MB`);
  }
  const buf = new Uint8Array(got); let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

// ---------- CNN (forward pass in cnn.js) ----------
async function loadCNN() {
  setStatus('CNN', 'loading…');
  const base = './models/cnn/';
  const [cfg, vocab, manifest] = await Promise.all(['config.json', 'vocab.json', 'manifest.json']
    .map((f) => fetch(base + f).then((r) => { if (!r.ok) throw new Error(f); return r.json(); })));
  const [lex, emo] = await Promise.all(['vader/vader_lexicon.txt', 'vader/emoji_utf8_lexicon.txt']
    .map((f) => fetch(f).then((r) => r.text())));
  sentiment = createVader(lex, emo);
  const bytes = await fetchWithProgress(base + 'weights.bin', 'CNN');
  const model = { W: unpackWeights(new Float32Array(bytes.buffer), manifest), manifest, cfg, vocab, sentiment };
  MODELS.CNN = { prob: async (text, title, rating) => cnnProb(model, text, title, rating), threshold: cfg.threshold,
                 explain: (text, title, rating) => cnnExplain(model, text, title, rating) };
  setStatus('CNN', 'ready');
}

// ---------- BERT / XLNet (ONNX int8) ----------
async function loadTransformer(name, dir) {
  setStatus(name, 'loading…');
  const cfg = await fetch(`./models/${dir}/config.json`).then((r) => r.json());
  const [tkJson, tkConfig] = await Promise.all(['tokenizer.json', 'tokenizer_config.json']
    .map((f) => fetch(`./models/${dir}/${f}`).then((r) => r.json())));
  const tokenizer = new TOKENIZER_CLASS[dir](tkJson, tkConfig);
  const bytes = await fetchWithProgress(`./models/${dir}/model.onnx`, name);
  setStatus(name, 'starting…');
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  const prob = async (text, title) => {
    const enc = await tokenizer(title + ' . ' + text, { truncation: true, max_length: cfg.max_len });
    const ids = enc.input_ids.data, n = ids.length;
    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, n]),
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(enc.attention_mask.data, BigInt), [1, n]),
      token_type_ids: new ort.Tensor('int64', new BigInt64Array(n), [1, n]),
    };
    const inputs = Object.fromEntries(session.inputNames.map((k) => [k, feeds[k]]));
    const logits = (await session.run(inputs))[session.outputNames[0]].data;
    const m = Math.max(logits[0], logits[1]), e0 = Math.exp(logits[0] - m), e1 = Math.exp(logits[1] - m);
    return e1 / (e0 + e1);
  };
  MODELS[name] = { prob, threshold: cfg.threshold };
  setStatus(name, 'ready');
}

// ---------- UI ----------
let rating = 5;
function drawStars() {
  $('stars').innerHTML = '';
  for (let i = 1; i <= 5; i++) {
    const b = document.createElement('button');
    b.type = 'button'; b.textContent = '★'; b.className = i <= rating ? 'on' : '';
    b.setAttribute('aria-label', `${i} star${i > 1 ? 's' : ''}`);
    b.onclick = () => { rating = i; drawStars(); };
    $('stars').appendChild(b);
  }
}
drawStars();

const EXAMPLES = [
  ['Absolutely amazing product!!! Best purchase ever, everyone should buy this, five stars!!!', 'Amazing', 5],
  ['I have been using this charger for about three months with my phone and tablet. It charges noticeably faster than the original one, the cable is thick and has not frayed, and the brick stays only slightly warm. The only downside is that the LED is very bright at night, so I keep it covered. For the price it has been reliable and I would buy it again.', 'Reliable fast charger after 3 months', 4],
  ['Stopped working after a week. Waste of money.', 'Broke quickly', 1],
];
document.querySelectorAll('[data-ex]').forEach((a) => (a.onclick = () => {
  const [t, s, r] = EXAMPLES[+a.dataset.ex]; $('review').value = t; $('title').value = s; rating = r; drawStars();
}));
$('clear').onclick = () => { $('review').value = ''; $('title').value = ''; rating = 5; drawStars(); $('why').hidden = true; };

function updateButton() {
  const ready = Object.keys(MODELS).length;
  $('check').disabled = !ready;
  $('check').textContent = ready ? 'Check review' : 'Loading models…';
}

$('check').onclick = async () => {
  const text = $('review').value.trim(), title = $('title').value.trim();
  if (!text) { $('review').focus(); return; }
  $('check').disabled = true; $('check').textContent = 'Checking…';
  try {
    const results = [];
    for (const name of ['CNN', 'BERT', 'XLNet']) {
      if (!MODELS[name]) continue;
      const p = await MODELS[name].prob(text, title, rating);
      results.push({ name, p, th: CUTOFF, fake: p >= CUTOFF });
    }
    // Final verdict: average fake probability of the loaded models, FAKE at 50% or more.
    const avg = results.reduce((s, r) => s + r.p, 0) / results.length;
    const fake = avg >= CUTOFF;
    const v = $('verdict');
    v.className = 'verdict ' + (fake ? 'fake' : 'real');
    v.innerHTML = `<div class="word">${fake ? 'FAKE' : 'GENUINE'}</div>
      <div class="sub">${shownPct(avg)} (average of ${results.map((r) => r.name).join(', ')})</div>`;
    // Each model shows the percentage for the label it gives: fake % for FAKE, genuine % for GENUINE.
    $('models').innerHTML = results.map((r) => `
      <div class="model">
        <div class="row"><span>${r.name}</span>
          <span class="tag ${r.fake ? 'fake' : 'real'}">${r.fake ? 'FAKE' : 'GENUINE'} · ${shownPct(r.p)}</span></div>
        <div class="track"><div class="fill ${r.fake ? 'fake' : 'real'}" style="width:${(labelProb(r.p) * 100).toFixed(1)}%"></div>
          <div class="thr" style="left:50%" title="50% line"></div></div>
      </div>`).join('');
    window.lastResults = results;   // handy for checking against the Python notebook
    renderWhy(text, title, rating, results, avg, fake);
  } catch (e) {
    $('verdict').innerHTML = `<div class="sub">Error: ${e.message}</div>`;
    console.error(e);
  } finally { updateButton(); }
};

// ---------- "Why this result?" ----------
// Typical values in the 291,762 training reviews (medians by label), used to describe each feature.
const TYPICAL = {
  review_length: { fake: 102, genuine: 10 },
  review_sentiment: { fake: 0.90, genuine: 0.61 },
  summary_sentiment: { fake: 0.27, genuine: 0.00 },
  rating_sentiment_difference: { fake: 0.01, genuine: 0.17 },
};
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const tone = (s) => (s >= 0.6 ? 'very positive' : s >= 0.05 ? 'positive' : s > -0.05 ? 'neutral' : s > -0.6 ? 'negative' : 'very negative');

function describeFeature({ name, value }) {
  const t = TYPICAL[name];
  switch (name) {
    case 'review_length':
      return `Review length: <b>${value} words</b>. In the training data, fake-labelled reviews tended to be longer (median ${t.fake} words vs ${t.genuine} for genuine ones), so length nudges the score. It is only one signal: short reviews can be fake and long ones genuine.`;
    case 'review_sentiment':
      return `Tone of the review: <b>${tone(value)}</b> (sentiment ${value.toFixed(2)} on a −1…+1 scale). In the training data, fake-labelled reviews tended to sound more positive (median ${t.fake.toFixed(2)} vs ${t.genuine.toFixed(2)}), but plenty of genuine reviews are very positive too.`;
    case 'summary_sentiment':
      return `Tone of the title: <b>${tone(value)}</b> (${value.toFixed(2)}). In the training data, fake-labelled titles leaned slightly more positive (median ${t.fake.toFixed(2)} vs ${t.genuine.toFixed(2)}); this is a weak signal on its own.`;
    case 'rating':
      return `Star rating: <b>${value}★</b>. Ratings on their own separate fake and genuine reviews only weakly; they matter mostly together with the tone of the text.`;
    case 'rating_sentiment_difference':
      return `Match between stars and tone: gap <b>${value.toFixed(2)}</b>. In the training data, stars and wording matched more closely in fake-labelled reviews (median gap ${t.fake.toFixed(2)} vs ${t.genuine.toFixed(2)}); a close match is common in genuine reviews as well.`;
    default:
      return name;
  }
}

function renderWhy(text, title, rating, results, avg, fake) {
  const box = $('why');
  if (!MODELS.CNN?.explain) { box.hidden = true; return; }
  const ex = MODELS.CNN.explain(text, title, rating);
  const pts = (d) => `${d > 0 ? '+' : '−'}${Math.abs(d * 100).toFixed(1)} pts`;
  const dirOf = (d) => (Math.abs(d) < 0.005 ? ['neutral', 'little effect'] : d > 0 ? ['fake', `▲ towards FAKE`] : ['real', `▼ towards GENUINE`]);

  // Reasons: engineered features ranked by influence, then the most influential words.
  const feats = ex.features.slice().sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  const words = ex.words.filter((w) => /\w/.test(w.tok));
  const topFake = words.filter((w) => w.delta > 0.002).sort((a, b) => b.delta - a.delta).slice(0, 5);
  const topReal = words.filter((w) => w.delta < -0.002).sort((a, b) => a.delta - b.delta).slice(0, 5);
  const items = feats.map((f) => {
    const [cls, label] = dirOf(f.delta);
    return `<li><span class="dir ${cls}">${label}<br><small>${pts(f.delta)}</small></span><span>${describeFeature(f)}</span></li>`;
  });
  if (topFake.length) items.push(`<li><span class="dir fake">▲ towards FAKE</span><span>Words that raised the fake score: ${topFake.map((w) => `<b>${escapeHtml(w.tok)}</b>`).join(', ')}</span></li>`);
  if (topReal.length) items.push(`<li><span class="dir real">▼ towards GENUINE</span><span>Words that lowered the fake score: ${topReal.map((w) => `<b>${escapeHtml(w.tok)}</b>`).join(', ')}</span></li>`);
  $('why-reasons').innerHTML = items.join('');

  // Summary sentence.
  const leanFake = results.filter((r) => r.fake).length;
  const main = feats.find((f) => Math.abs(f.delta) >= 0.005 && (f.delta > 0) === fake);
  const mainText = main ? {
    review_length: `its length (${main.value} words)`, review_sentiment: `its ${tone(main.value)} tone`,
    summary_sentiment: `the ${tone(main.value)} title`, rating: `the ${main.value}★ rating`,
    rating_sentiment_difference: 'how closely the stars match the wording',
  }[main.name] : null;
  $('why-summary').innerHTML = `<b style="color:var(--${fake ? 'fake' : 'real'})">${fake ? 'FAKE' : 'GENUINE'}</b> — ${shownPct(avg)} on average across the models. ${leanFake} of ${results.length} model${results.length > 1 ? 's' : ''} lean${results.length === 1 ? 's' : ''} fake.
    ${mainText ? `The biggest factor for the CNN was ${mainText}.` : ''}`;

  // Highlighted text (tokens as the CNN sees them: lowercased title + review).
  const maxAbs = Math.max(1e-6, ...ex.words.map((w) => Math.abs(w.delta)));
  $('why-text').innerHTML = ex.words.map((w) => {
    const a = Math.min(1, Math.abs(w.delta) / maxAbs);
    if (a < 0.08) return escapeHtml(w.tok);
    const rgb = w.delta > 0 ? '198,40,40' : '31,122,58';
    return `<span class="hl" style="background:rgba(${rgb},${(0.15 + 0.6 * a).toFixed(2)})" title="${pts(w.delta)}">${escapeHtml(w.tok)}</span>`;
  }).join(' ');
  box.hidden = false;
}

// Load the small CNN first so the page is usable quickly, then the transformers.
(async () => {
  for (const [name, fn] of [['CNN', loadCNN], ['BERT', () => loadTransformer('BERT', 'bert')],
                            ['XLNet', () => loadTransformer('XLNet', 'xlnet')]]) {
    try { await fn(); } catch (e) { setStatus(name, 'not available'); console.error(name, e); }
    updateButton();
  }
})();
