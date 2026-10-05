// Line-by-line port of vaderSentiment 3.3.2 (C.J. Hutto, MIT licence) — the exact version used to build the
// training features — so the CNN gets the same sentiment inputs in the browser as in Python.
// Only polarity "compound" is needed by the CNN; pos/neu/neg are omitted.

const B_INCR = 0.293, B_DECR = -0.293, C_INCR = 0.733, N_SCALAR = -0.74;

const NEGATE = new Set(['aint', 'arent', 'cannot', 'cant', 'couldnt', 'darent', 'didnt', 'doesnt',
  "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
  'dont', 'hadnt', 'hasnt', 'havent', 'isnt', 'mightnt', 'mustnt', 'neither',
  "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
  'neednt', "needn't", 'never', 'none', 'nope', 'nor', 'not', 'nothing', 'nowhere',
  'oughtnt', 'shant', 'shouldnt', 'uhuh', 'wasnt', 'werent',
  "oughtn't", "shan't", "shouldn't", 'uh-uh', "wasn't", "weren't",
  'without', 'wont', 'wouldnt', "won't", "wouldn't", 'rarely', 'seldom', 'despite']);

const BOOSTER = {};
for (const w of ['absolutely', 'amazingly', 'awfully', 'completely', 'considerable', 'considerably',
  'decidedly', 'deeply', 'effing', 'enormous', 'enormously', 'entirely', 'especially', 'exceptional',
  'exceptionally', 'extreme', 'extremely', 'fabulously', 'flipping', 'flippin', 'frackin', 'fracking',
  'fricking', 'frickin', 'frigging', 'friggin', 'fully', 'fuckin', 'fucking', 'fuggin', 'fugging',
  'greatly', 'hella', 'highly', 'hugely', 'incredible', 'incredibly', 'intensely', 'major', 'majorly',
  'more', 'most', 'particularly', 'purely', 'quite', 'really', 'remarkably', 'so', 'substantially',
  'thoroughly', 'total', 'totally', 'tremendous', 'tremendously', 'uber', 'unbelievably', 'unusually',
  'utter', 'utterly', 'very']) BOOSTER[w] = B_INCR;
for (const w of ['almost', 'barely', 'hardly', 'just enough', 'kind of', 'kinda', 'kindof', 'kind-of',
  'less', 'little', 'marginal', 'marginally', 'occasional', 'occasionally', 'partly', 'scarce',
  'scarcely', 'slight', 'slightly', 'somewhat', 'sort of', 'sorta', 'sortof', 'sort-of']) BOOSTER[w] = B_DECR;
const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

const SPECIAL_CASES = { 'the shit': 3, 'the bomb': 3, 'bad ass': 1.5, 'badass': 1.5, 'bus stop': 0.0,
  'yeah right': -2, 'kiss of death': -1.5, 'to die for': 3, 'beating heart': 3.5 };

const PUNCT = new Set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~');
const isUpper = (s) => s.toLowerCase() !== s.toUpperCase() && s === s.toUpperCase();   // Python str.isupper
const lower = (s) => s.toLowerCase();

function negated(words) {
  for (const w of words.map(lower)) if (NEGATE.has(w)) return true;
  for (const w of words) if (w.includes("n't")) return true;
  return false;
}
function normalize(score, alpha = 15) {
  const n = score / Math.sqrt(score * score + alpha);
  return n < -1 ? -1 : n > 1 ? 1 : n;
}
function allcapDifferential(words) {
  let allcap = 0;
  for (const w of words) if (isUpper(w)) allcap++;
  const diff = words.length - allcap;
  return diff > 0 && diff < words.length;
}
function scalarIncDec(word, valence, isCapDiff) {
  let scalar = 0.0;
  const wl = lower(word);
  if (hasOwn(BOOSTER, wl)) {
    scalar = BOOSTER[wl];
    if (valence < 0) scalar *= -1;
    if (isUpper(word) && isCapDiff) scalar += valence > 0 ? C_INCR : -C_INCR;
  }
  return scalar;
}
function stripPunc(token) {
  const cps = [...token];
  let a = 0, b = cps.length;
  while (a < b && PUNCT.has(cps[a])) a++;
  while (b > a && PUNCT.has(cps[b - 1])) b--;
  return b - a <= 2 ? token : cps.slice(a, b).join('');
}

/** Build an analyzer from the text of vader_lexicon.txt and emoji_utf8_lexicon.txt. Returns text => compound. */
export function createVader(lexiconText, emojiText) {
  const lexicon = {};
  for (const line of lexiconText.replace(/\n+$/, '').split('\n')) {
    if (!line) continue;
    const [w, m] = line.trim().split('\t');
    lexicon[w] = parseFloat(m);
  }
  const emojis = new Map();
  for (const line of emojiText.replace(/\n+$/, '').split('\n')) {
    const [e, d] = line.trim().split('\t');
    emojis.set(e, d);
  }
  const inLex = (w) => hasOwn(lexicon, w);

  function negationCheck(valence, wl, startI, i) {
    if (startI === 0) {
      if (negated([wl[i - 1]])) valence *= N_SCALAR;
    }
    if (startI === 1) {
      if (wl[i - 2] === 'never' && (wl[i - 1] === 'so' || wl[i - 1] === 'this')) valence *= 1.25;
      else if (wl[i - 2] === 'without' && wl[i - 1] === 'doubt') { /* unchanged */ }
      else if (negated([wl[i - 2]])) valence *= N_SCALAR;
    }
    if (startI === 2) {
      if ((wl[i - 3] === 'never' && (wl[i - 2] === 'so' || wl[i - 2] === 'this')) ||
          (wl[i - 1] === 'so' || wl[i - 1] === 'this')) valence *= 1.25;
      else if (wl[i - 3] === 'without' && (wl[i - 2] === 'doubt' || wl[i - 1] === 'doubt')) { /* unchanged */ }
      else if (negated([wl[i - 3]])) valence *= N_SCALAR;
    }
    return valence;
  }

  function specialIdiomsCheck(valence, wl, i) {
    const at = (k) => wl[k < 0 ? wl.length + k : k];        // Python negative indexing
    const onezero = `${at(i - 1)} ${at(i)}`, twoonezero = `${at(i - 2)} ${at(i - 1)} ${at(i)}`;
    const twoone = `${at(i - 2)} ${at(i - 1)}`, threetwoone = `${at(i - 3)} ${at(i - 2)} ${at(i - 1)}`;
    const threetwo = `${at(i - 3)} ${at(i - 2)}`;
    for (const seq of [onezero, twoonezero, twoone, threetwoone, threetwo])
      if (hasOwn(SPECIAL_CASES, seq)) { valence = SPECIAL_CASES[seq]; break; }
    if (wl.length - 1 > i) {
      const zeroone = `${wl[i]} ${wl[i + 1]}`;
      if (hasOwn(SPECIAL_CASES, zeroone)) valence = SPECIAL_CASES[zeroone];
    }
    if (wl.length - 1 > i + 1) {
      const zeroonetwo = `${wl[i]} ${wl[i + 1]} ${wl[i + 2]}`;
      if (hasOwn(SPECIAL_CASES, zeroonetwo)) valence = SPECIAL_CASES[zeroonetwo];
    }
    for (const ng of [threetwoone, threetwo, twoone]) if (hasOwn(BOOSTER, ng)) valence += BOOSTER[ng];
    return valence;
  }

  function leastCheck(valence, wl, i) {
    if (i > 1 && !inLex(wl[i - 1]) && wl[i - 1] === 'least') {
      if (wl[i - 2] !== 'at' && wl[i - 2] !== 'very') valence *= N_SCALAR;
    } else if (i > 0 && !inLex(wl[i - 1]) && wl[i - 1] === 'least') {
      valence *= N_SCALAR;
    }
    return valence;
  }

  function sentimentValence(words, wl, isCapDiff, item, i, sentiments) {
    let valence = 0;
    const il = lower(item);
    if (inLex(il)) {
      valence = lexicon[il];
      if (il === 'no' && i !== words.length - 1 && inLex(wl[i + 1])) valence = 0.0;
      if ((i > 0 && wl[i - 1] === 'no') || (i > 1 && wl[i - 2] === 'no') ||
          (i > 2 && wl[i - 3] === 'no' && (wl[i - 1] === 'or' || wl[i - 1] === 'nor')))
        valence = lexicon[il] * N_SCALAR;
      if (isUpper(item) && isCapDiff) valence += valence > 0 ? C_INCR : -C_INCR;
      for (let startI = 0; startI < 3; startI++) {
        if (i > startI && !inLex(wl[i - (startI + 1)])) {
          let s = scalarIncDec(words[i - (startI + 1)], valence, isCapDiff);
          if (startI === 1 && s !== 0) s *= 0.95;
          if (startI === 2 && s !== 0) s *= 0.9;
          valence += s;
          valence = negationCheck(valence, wl, startI, i);
          if (startI === 2) valence = specialIdiomsCheck(valence, wl, i);
        }
      }
      valence = leastCheck(valence, wl, i);
    }
    sentiments.push(valence);
  }

  function butCheck(wl, sentiments) {
    const bi = wl.indexOf('but');
    if (bi >= 0) {
      // Python iterates the list while replacing items found via list.index(value); replicate exactly.
      for (let p = 0; p < sentiments.length; p++) {
        const s = sentiments[p];
        const si = sentiments.indexOf(s);
        if (si < bi) sentiments[si] = s * 0.5;
        else if (si > bi) sentiments[si] = s * 1.5;
      }
    }
    return sentiments;
  }

  return function compound(rawText) {
    let text = '', prevSpace = true;
    for (const ch of String(rawText)) {
      if (emojis.has(ch)) {
        if (!prevSpace) text += ' ';
        text += emojis.get(ch);
        prevSpace = false;
      } else {
        text += ch;
        prevSpace = ch === ' ';
      }
    }
    text = text.trim();
    const words = text.split(/\s+/).filter(Boolean).map(stripPunc);
    const wl = words.map(lower);
    const isCapDiff = allcapDifferential(words);

    const sentiments = [];
    words.forEach((item, i) => {
      if (hasOwn(BOOSTER, wl[i])) { sentiments.push(0); return; }
      if (i < words.length - 1 && wl[i] === 'kind' && wl[i + 1] === 'of') { sentiments.push(0); return; }
      sentimentValence(words, wl, isCapDiff, item, i, sentiments);
    });
    butCheck(wl, sentiments);

    if (!sentiments.length) return 0.0;
    let sum = sentiments.reduce((a, b) => a + b, 0);
    const ep = Math.min((text.match(/!/g) || []).length, 4) * 0.292;
    const qc = (text.match(/\?/g) || []).length;
    const qm = qc > 1 ? (qc <= 3 ? qc * 0.18 : 0.96) : 0;
    if (sum > 0) sum += ep + qm; else if (sum < 0) sum -= ep + qm;
    return Number(normalize(sum).toFixed(4));
  };
}
