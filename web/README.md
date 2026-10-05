---
title: Fake Review Checker
emoji: 🔍
colorFrom: indigo
colorTo: red
sdk: static
app_file: index.html
pinned: false
---

# Fake Review Checker

Classifies an Amazon product review as **FAKE** or **GENUINE** with three models trained on 291,762 reviews:
a CNN (GloVe + Conv1D + HACE loss), fine-tuned BERT and fine-tuned XLNet. The models run in the visitor's
browser (ONNX Runtime Web + transformers.js tokenizers), so the site is fully static.

Code: https://github.com/AjayRaju-18/ShreyaProject
