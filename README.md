# Stack Overflow tag prediction from titles: ML vs DL PoC

Proof of concept: given only the **title** of a Stack Overflow post, predict its **tags**, and compare two approaches on the same data and metrics.

| | Approach |
|---|---|
| **ML model** | TF-IDF (word + character n-grams) → One-vs-Rest logistic regression (scikit-learn) |
| **DL model** | Fine-tuned `distilbert-base-uncased` with a multi-label sigmoid head (Hugging Face `transformers`, PyTorch) |

Domain vocabulary (post, title, tag, kept tag, label set, …) is defined in [`GLOSSARY.md`](GLOSSARY.md).

## Run it

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lrondal/poc-nlp/blob/main/notebooks/stackoverflow_tag_prediction.ipynb)

1. Open [`notebooks/stackoverflow_tag_prediction.ipynb`](notebooks/stackoverflow_tag_prediction.ipynb) in Google Colab (badge above).
2. *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all*. The first cell clones this repository to get the data; everything else is preinstalled on Colab.

The ML part also runs on any laptop CPU in about 30 seconds; the DL part needs a GPU to be practical.

## Method

1. **Data**: `data/stackoverflow_posts.csv`, one row per *(post, tag)*: 4,609 posts, 682 distinct tags, 1–5 tags per post.
2. **Kept tags**: only tags present on ≥ 20 posts are predicted (86 tags); posts left without any kept tag are dropped (4,100 posts remain).
3. **Split**: by post, 80 / 10 / 10 train / validation / test, fixed seed, identical for both models.
4. **Decision rule**: each model scores every kept tag in [0, 1]; predicted tags are those above a threshold tuned on validation (max micro-F1), with at least the top tag.
5. **Metrics on test**: micro-F1 (headline), macro-F1, precision@1, recall@3, training time.

## Results

| Model | micro-F1 | macro-F1 | P@1 | R@3 |
|---|---|---|---|---|
| ML model (TF-IDF + LogReg) | 0.532 | 0.427 | 0.627 | 0.665 |
| DL model (DistilBERT) | *run on Colab* | | | |

ML figures come from a local CPU run. Fill in the DL row (and the conclusion in the notebook) after the Colab run.

## Data

Stack Overflow content is licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/) by Stack Exchange, Inc. and its contributors.
