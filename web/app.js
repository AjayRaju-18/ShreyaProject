// Fake Review Checker — runs the CNN, BERT and XLNet models entirely in the browser.
// Model files live in ./models/ (uploaded by notebooks/06_deploy_web.ipynb).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';
import { cnnProb, unpackWeights } from './cnn.js';
import { createVader } from './vader.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL('./models/', location.href).href;

const BEST_ORDER = ['XLNet', 'BERT', 'CNN'];
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
  MODELS.CNN = { prob: async (text, title, rating) => cnnProb(model, text, title, rating), threshold: cfg.threshold };
  setStatus('CNN', 'ready');
}

// ---------- BERT / XLNet (ONNX int8) ----------
async function loadTransformer(name, dir) {
  setStatus(name, 'loading…');
  const cfg = await fetch(`./models/${dir}/config.json`).then((r) => r.json());
  const tokenizer = await AutoTokenizer.from_pretrained(dir);
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
$('clear').onclick = () => { $('review').value = ''; $('title').value = ''; rating = 5; drawStars(); };

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
      results.push({ name, p, th: MODELS[name].threshold, fake: p >= MODELS[name].threshold });
    }
    const best = BEST_ORDER.map((n) => results.find((r) => r.name === n)).find(Boolean);
    const v = $('verdict');
    v.className = 'verdict ' + (best.fake ? 'fake' : 'real');
    v.innerHTML = `<div class="word">${best.fake ? 'FAKE' : 'GENUINE'}</div>
      <div class="sub">Final verdict from ${best.name} · fake probability ${(best.p * 100).toFixed(1)}%</div>`;
    $('models').innerHTML = results.map((r) => `
      <div class="model">
        <div class="row"><span>${r.name}</span>
          <span class="tag ${r.fake ? 'fake' : 'real'}">${r.fake ? 'FAKE' : 'GENUINE'} · ${(r.p * 100).toFixed(1)}%</span></div>
        <div class="track"><div class="fill ${r.fake ? 'fake' : 'real'}" style="width:${(r.p * 100).toFixed(1)}%"></div>
          <div class="thr" style="left:${(r.th * 100).toFixed(1)}%" title="threshold ${r.th.toFixed(2)}"></div></div>
      </div>`).join('');
    window.lastResults = results;   // handy for checking against the Python notebook
  } catch (e) {
    $('verdict').innerHTML = `<div class="sub">Error: ${e.message}</div>`;
    console.error(e);
  } finally { updateButton(); }
};

// Load the small CNN first so the page is usable quickly, then the transformers.
(async () => {
  for (const [name, fn] of [['CNN', loadCNN], ['BERT', () => loadTransformer('BERT', 'bert')],
                            ['XLNet', () => loadTransformer('XLNet', 'xlnet')]]) {
    try { await fn(); } catch (e) { setStatus(name, 'not available'); console.error(name, e); }
    updateButton();
  }
})();
