# Fake Review Detection Using Deep Learning and Transformer-Based Models

Department of Information Technology, Government College of Technology
Team: Anulakshmi A · Shreya M · Tamilarasi T · Navyaa Sharma — Guide: Dr. R. Malavika

This project classifies Amazon product reviews as **Fake (1)** or **Genuine (0)** using a CNN, BERT and XLNet, then compares the three models.

## Notebooks (run in Google Colab, GPU runtime)

| # | Notebook | What it does |
|---|----------|--------------|
| 01 | [`notebooks/01_cnn_fake_review.ipynb`](notebooks/01_cnn_fake_review.ipynb) | CNN from Jayasinghe & Dassanayaka (2025): GloVe 100d + 3×Conv1D (128/64/32, k=5) + engineered features, trained with the HACE loss |
| 02 | [`notebooks/02_bert_fake_review.ipynb`](notebooks/02_bert_fake_review.ipynb) | Fine-tunes `bert-base-uncased` |
| 03 | [`notebooks/03_xlnet_fake_review.ipynb`](notebooks/03_xlnet_fake_review.ipynb) | Fine-tunes `xlnet-base-cased` |
| 04 | [`notebooks/04_model_comparison.ipynb`](notebooks/04_model_comparison.ipynb) | Comparison table, bar chart, ROC/PR curves, confusion matrices, per-category F1 |
| 05 | [`notebooks/05_predict.ipynb`](notebooks/05_predict.ipynb) | Loads the trained models from Drive and classifies new reviews (single reviews or a whole CSV) without retraining |

## Trained models (in Drive `outputs/`)

| Model | Files |
|---|---|
| CNN | `cnn_model.keras`, `cnn_vocab.json`, `cnn_scaler.pkl`, `cnn_config.json` |
| BERT | `bert_model/` (weights, tokenizer, `predict_config.json`) |
| XLNet | `xlnet_model/` (weights, tokenizer, `predict_config.json`) |

To use them, open `05_predict.ipynb` in Colab, run all cells, and call `predict([...reviews...])`.

Open any notebook in Colab via **File → Open notebook → GitHub** and paste this repository's URL.

## Dataset setup (one time)

The dataset is too large for GitHub (`train.csv` is about 100 MB), so it is kept in Google Drive:

```
MyDrive/FakeReviewProject/
├── dataset/
│   ├── train.csv   (291,762 reviews: 266,227 genuine / 25,535 fake)
│   └── test.csv    (125,042 reviews: 114,099 genuine / 10,943 fake)
└── outputs/        (created automatically: metrics, plots, saved models)
```

Teammates: add the shared `FakeReviewProject` folder to your own Drive (*Organize → Add shortcut to Drive → My Drive*). If you use a different path, change `PROJECT_DIR` in the first cell of each notebook.

## Run order

1. `01_cnn` (~15–25 min on a T4)
2. `02_bert` (~45–60 min)
3. `03_xlnet` (~60–90 min)
4. `04_model_comparison`

Each notebook writes `outputs/<model>_metrics.json` and `outputs/<model>_test_probs.npy`, which notebook 04 reads.

## Method summary

- **Labels** (paper §3.2): a reviewer whose non-verified-purchase ratio is above 0.5 is treated as fake, and all of that reviewer's reviews are labelled fake.
- **Features** (paper Table 1): review length, review sentiment (VADER), summary sentiment, rating, rating–sentiment difference.
- **CNN input**: summary + review text, tokenised with NLTK `TweetTokenizer`, padded to 200 tokens, plus the scaled engineered features.
- **Imbalance**: HACE loss for the CNN (α = 0.1, γ = 2, class weight recomputed every epoch); class-weighted cross-entropy for BERT and XLNet.
- **Transformers** are fine-tuned on a stratified 30k subsample of the training set (2 epochs, max length 256, learning rate 2e-5).
- **Evaluation**: all models are tested on the same full test set. The decision threshold is chosen on a validation split taken from the training data.
- **Metrics**: accuracy, precision, recall, F1-Fake, macro-F1, ROC-AUC, confusion matrix, and results per product category.

> Note: `nvp_ratio`, `nvp_reviews` and `suspicious_reviewer` are deliberately **not** used as model inputs. The label is defined from them, so using them would leak the answer.

## References

- Jayasinghe, J. M. T., & Dassanayaka, S. (2025). Detecting deception: employing deep neural networks for fraudulent review detection on Amazon. *Neural Computing and Applications*, 37, 21715–21742.
- Tao, J., Fang, X., & Zhou, L. (2026). Toward a Knowledge Discovery Method to Fake Review Detection. *Information Systems Frontiers*, 28, 1109–1125.
- Ni, J., Li, J., & McAuley, J. (2019). Amazon Review Data (2018).

## Fake Review Checker app

Run all cells of `notebooks/05_predict.ipynb` in Colab. The last cell opens an app: type a review, click **Check review**, and it shows **FAKE / GENUINE** with each model's fake probability. Set `SHARE = True` in that cell to get a temporary public link for teammates.

## Website (Hugging Face Space)

`space/` holds the website (`app.py`). Run `notebooks/06_deploy_space.ipynb` in Colab to publish it, together with the trained models from Drive, to `https://huggingface.co/spaces/<username>/fake-review-checker`. To remove the website, delete the Space on Hugging Face and the `space/` folder here.
