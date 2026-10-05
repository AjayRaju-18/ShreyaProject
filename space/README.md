---
title: Fake Review Checker
emoji: 🔍
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
pinned: false
---

# Fake Review Checker

Classifies an Amazon product review as **FAKE** or **GENUINE** using three models trained on 291,762 reviews:
a CNN (GloVe + Conv1D + HACE loss), fine-tuned BERT and fine-tuned XLNet.

Test-set results (125,042 reviews):

| Model | Accuracy | F1-Fake | ROC-AUC |
|---|---|---|---|
| CNN | 0.930 | 0.569 | 0.885 |
| BERT | 0.930 | 0.580 | 0.884 |
| XLNet | 0.932 | 0.583 | 0.887 |

Code: https://github.com/AjayRaju-18/ShreyaProject
