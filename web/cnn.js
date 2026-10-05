// CNN from notebook 01 (Jayasinghe & Dassanayaka 2025), forward pass in plain JavaScript.
// clean -> TweetTokenizer -> ids -> Embedding -> Conv1D(128)+pool -> Conv1D(64)+pool -> Conv1D(32)
// -> GlobalMaxPool -> concat scaled engineered features -> Dense(64, relu) -> Dense(1, sigmoid)

const URL_RE = /https?:\/\/\S+|www\.\S+/g, HTML_RE = /<[^>]+>/g, NONTXT_RE = /[^a-z0-9'!?.,:;()\s]/g;
export const cleanText = (t) =>
  String(t).toLowerCase().replace(URL_RE, ' ').replace(HTML_RE, ' ').replace(NONTXT_RE, ' ');

// Port of NLTK TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True), in the same
// alternation order as nltk.tokenize.casual.REGEXPS_PHONE. Patterns for characters that cleanText()
// always removes (#, @, <, >, -, /, emoji) are omitted because they can never match.
const MOUTH = String.raw`[\)\]\(\[dDpP\/:\}\{@\|\\]`;
const TOKEN_RE = new RegExp([
  // naked domains such as amazon.com (NLTK URLS, second branch)
  String.raw`[a-z0-9]+(?:[.\-][a-z0-9]+){0,126}[.](?:[a-z]{2,13})\b\/?`,
  // phone numbers
  String.raw`(?:(?:\+?[01][ *\-.\)]*)?(?:[\(]?\d{3}[ *\-.\)]*)?\d{3}[ *\-.\)]*\d{4})`,
  // emoticons, both directions
  String.raw`(?:[<>]?[:;=8][\-o\*']?${MOUTH}|${MOUTH}[\-o\*']?[:;=8][<>]?)`,
  String.raw`(?:[a-z](?:[a-z]|['\-_])+[a-z])`,                   // words with apostrophes or dashes
  String.raw`(?:[+\-]?\d+[,\/.:\-]\d+[+\-]?)`,                    // numbers, fractions, decimals
  String.raw`(?:[\w_]+)`,                                         // other words
  String.raw`(?:\.(?:\s*\.){1,})`,                                // ellipsis
  String.raw`(?:\S)`,                                             // everything else
].join('|'), 'gi');
export const tweetTokenize = (t) =>
  (t.replace(/(.)\1{2,}/g, '$1$1$1')                // reduce_lengthening
    .replace(/([^\p{L}\p{N}])\1{3,}/gu, '$1$1$1')   // HANG_RE
    .match(TOKEN_RE) || []).map((w) => (/^(?:[<>]?[:;=8][\-o\*']?[\)\]\(\[dDpP\/:\}\{@\|\\]|[\)\]\(\[dDpP\/:\}\{@\|\\][\-o\*']?[:;=8][<>]?)$/i.test(w) ? w : w.toLowerCase()));

function conv1dRelu(x, len, cin, k, kernel, bias, cout) {
  const outLen = len - k + 1, out = new Float32Array(outLen * cout);
  for (let t = 0; t < outLen; t++) {
    const o = t * cout;
    for (let f = 0; f < cout; f++) out[o + f] = bias[f];
    for (let j = 0; j < k; j++) {
      const xi = (t + j) * cin, kj = j * cin * cout;
      for (let c = 0; c < cin; c++) {
        const v = x[xi + c];
        if (v === 0) continue;
        const kc = kj + c * cout;
        for (let f = 0; f < cout; f++) out[o + f] += v * kernel[kc + f];
      }
    }
    for (let f = 0; f < cout; f++) if (out[o + f] < 0) out[o + f] = 0;
  }
  return [out, outLen];
}

function maxPool2(x, len, ch) {
  const outLen = Math.floor(len / 2), out = new Float32Array(outLen * ch);
  for (let t = 0; t < outLen; t++)
    for (let c = 0; c < ch; c++) out[t * ch + c] = Math.max(x[2 * t * ch + c], x[(2 * t + 1) * ch + c]);
  return [out, outLen];
}

function dense(x, kernel, bias, nout) {
  const out = Float32Array.from(bias);
  for (let i = 0; i < x.length; i++) {
    const v = x[i], r = i * nout;
    for (let o = 0; o < nout; o++) out[o] += v * kernel[r + o];
  }
  return out;
}

/** Slice the flat weights buffer into named tensors using manifest {name: {offset, shape}}. */
export function unpackWeights(all, manifest) {
  const W = {};
  for (const [name, { offset, shape }] of Object.entries(manifest))
    W[name] = all.subarray(offset, offset + shape.reduce((a, b) => a * b, 1));
  return W;
}

/** Fake probability for one review. `sentiment(text)` must return the VADER compound score. */
export function cnnProb({ W, manifest, cfg, vocab, sentiment }, text, title, rating) {
  const S = (n) => manifest[n].shape;
  const L = cfg.max_len, D = S('embedding')[1];
  const ids = tweetTokenize(cleanText(title + ' . ' + text)).slice(0, L).map((w) => vocab[w] ?? 1);

  let x = new Float32Array(L * D);                  // padding positions use embedding row 0, as in training
  for (let t = 0; t < L; t++) {
    const id = ids[t] ?? 0;
    x.set(W.embedding.subarray(id * D, (id + 1) * D), t * D);
  }
  let len = L, ch = D;
  for (const [i, pool] of [[1, true], [2, true], [3, false]]) {
    const [k, , cout] = S(`conv${i}_kernel`);
    [x, len] = conv1dRelu(x, len, ch, k, W[`conv${i}_kernel`], W[`conv${i}_bias`], cout);
    ch = cout;
    if (pool) [x, len] = maxPool2(x, len, ch);
  }
  const g = new Float32Array(ch).fill(-Infinity);
  for (let t = 0; t < len; t++) for (let c = 0; c < ch; c++) g[c] = Math.max(g[c], x[t * ch + c]);

  const rs = sentiment(text);
  const feats = {
    rating, review_length: text.trim().split(/\s+/).filter(Boolean).length, review_sentiment: rs,
    summary_sentiment: sentiment(title), rating_sentiment_difference: rating / 5 - rs,
  };
  const num = cfg.num_features.map((f, i) => (feats[f] - cfg.scaler_mean[i]) / cfg.scaler_scale[i]);
  const h = dense(Float32Array.from([...g, ...num]), W.dense1_kernel, W.dense1_bias, S('dense1_kernel')[1]);
  for (let i = 0; i < h.length; i++) if (h[i] < 0) h[i] = 0;
  const z = dense(h, W.dense2_kernel, W.dense2_bias, 1)[0];
  return 1 / (1 + Math.exp(-z));
}
