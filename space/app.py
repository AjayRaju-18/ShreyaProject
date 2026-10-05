"""Fake Review Checker — Hugging Face Space.

Loads the CNN, BERT and XLNet models trained in notebooks 01-03 (uploaded to models/
by notebook 06) and classifies a review as FAKE or GENUINE.
"""
import json, os, re
import numpy as np
import pandas as pd
import gradio as gr
import torch
from nltk.tokenize import TweetTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
vader = SentimentIntensityAnalyzer()
MODELS = {}   # name -> (prob_fn(texts, summaries, ratings), threshold)

# ---- CNN (Keras) ----
if os.path.exists(f'{MODEL_DIR}/cnn_model.keras'):
    import keras
    cfg = json.load(open(f'{MODEL_DIR}/cnn_config.json'))
    vocab = json.load(open(f'{MODEL_DIR}/cnn_vocab.json'))
    sc = json.load(open(f'{MODEL_DIR}/cnn_scaler.json'))
    mean, scale = np.array(sc['mean']), np.array(sc['scale'])
    cnn = keras.models.load_model(f'{MODEL_DIR}/cnn_model.keras', compile=False)
    tok = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    URL, HTML, NONTXT = re.compile(r'https?://\S+|www\.\S+'), re.compile(r'<[^>]+>'), re.compile(r"[^a-z0-9'!?.,:;()\s]")

    def clean(t):
        return NONTXT.sub(' ', HTML.sub(' ', URL.sub(' ', str(t).lower())))

    def cnn_prob(texts, summaries, ratings):
        X = np.zeros((len(texts), cfg['max_len']), dtype='int32')
        for i, (t, s) in enumerate(zip(texts, summaries)):
            ids = [vocab.get(w, 1) for w in tok.tokenize(clean(s + ' . ' + t))][:cfg['max_len']]
            X[i, :len(ids)] = ids
        rs = np.array([vader.polarity_scores(t)['compound'] for t in texts])
        feats = pd.DataFrame({'rating': ratings, 'review_length': [len(t.split()) for t in texts],
                              'review_sentiment': rs,
                              'summary_sentiment': [vader.polarity_scores(s)['compound'] for s in summaries],
                              'rating_sentiment_difference': np.asarray(ratings) / 5 - rs})
        num = ((feats[cfg['num_features']].values - mean) / scale).astype('float32')
        return cnn.predict([X, num], verbose=0).ravel()

    MODELS['CNN'] = (cnn_prob, cfg['threshold'])

# ---- BERT / XLNet (PyTorch) ----
for name in ['BERT', 'XLNet']:
    d = f'{MODEL_DIR}/{name.lower()}_model'
    if not os.path.exists(d):
        continue
    tcfg = json.load(open(f'{d}/predict_config.json'))
    tk = AutoTokenizer.from_pretrained(d)
    mdl = AutoModelForSequenceClassification.from_pretrained(d).eval()

    def tf_prob(texts, summaries, ratings, tk=tk, mdl=mdl, L=tcfg['max_len']):
        enc = tk([s + ' . ' + t for t, s in zip(texts, summaries)], truncation=True,
                 max_length=L, padding=True, return_tensors='pt')
        with torch.no_grad():
            return torch.softmax(mdl(**enc).logits, -1)[:, 1].numpy()

    MODELS[name] = (tf_prob, tcfg['threshold'])

BEST = next((m for m in ['XLNet', 'BERT', 'CNN'] if m in MODELS), None)
print('Loaded models:', list(MODELS), '| best:', BEST)


def check_review(review, title, rating):
    if not review or not review.strip():
        return '<b>Please type a review first.</b>', None
    if not MODELS:
        return '<b>No trained models found in models/.</b>', None
    rows, verdict = [], None
    for name, (fn, th) in MODELS.items():
        p = float(fn([review], [title or ''], [float(rating)])[0])
        label = 'FAKE' if p >= th else 'GENUINE'
        rows.append({'Model': name, 'Prediction': label, 'Fake probability': f'{p:.1%}', 'Threshold': round(th, 2)})
        if name == BEST:
            verdict, best_p = label, p
    colour = '#c62828' if verdict == 'FAKE' else '#2e7d32'
    head = (f"<div style='font-size:2.4em;font-weight:700;color:{colour}'>{verdict}</div>"
            f"<div>Final verdict from <b>{BEST}</b> (best model) — fake probability {best_p:.0%}</div>")
    return head, pd.DataFrame(rows)


with gr.Blocks(title='Fake Review Checker') as demo:
    gr.Markdown('# Fake Review Checker\n'
                'Detects fake Amazon product reviews with a **CNN**, **BERT** and **XLNet**. '
                'Type a review and click **Check review**.')
    with gr.Row():
        with gr.Column():
            review = gr.Textbox(label='Review text', lines=7, placeholder='Paste or type a product review...')
            title = gr.Textbox(label='Review title (optional)')
            rating = gr.Slider(1, 5, value=5, step=1, label='Star rating')
            btn = gr.Button('Check review', variant='primary')
        with gr.Column():
            verdict = gr.HTML()
            table = gr.Dataframe(label='All models', interactive=False)
    gr.Examples([['Absolutely amazing product!!! Best purchase ever, everyone should buy this, five stars!!!', 'Amazing', 5],
                 ['Case fits my phone well. The buttons are a bit stiff but it has survived two drops so far.', 'Decent case', 4]],
                inputs=[review, title, rating])
    btn.click(check_review, [review, title, rating], [verdict, table])
    gr.Markdown('Final-year project — Department of Information Technology, Government College of Technology. '
                'Labels are based on reviewer behaviour (non-verified-purchase ratio), so predictions reflect that definition of "fake".')

if __name__ == '__main__':
    demo.launch()
