# Cellula NLP Internship - Week 1: Toxic Comment Classification

## Project

Multi-label classification of text comments (Jigsaw Toxic Comment Classification dataset) across 6 categories simultaneously:
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

Two separate models were trained:
- **RNN** (Bidirectional RNN) - in the `RNN/` folder
- **LSTM** (Bidirectional LSTM) - in the `LSTM/` folder

## Pipeline

1. **Train/test split** (80/20) performed before any data modification, to ensure an unbiased evaluation
2. **Handling class imbalance** for the rare labels (threat, severe_toxic, identity_hate make up less than 3% of the data):
   - Synonym replacement augmentation (WordNet)
   - LLM few-shot augmentation (optional, requires `ANTHROPIC_API_KEY`)
   - Undersampling of "clean" comments (3:1 ratio against comments containing any toxicity)
3. **Text cleaning**: removing URLs, IP addresses, and non-English characters
4. **Word representation**: pretrained GloVe embeddings (100d) instead of training embeddings from scratch
5. **Model architecture**: Bidirectional RNN/LSTM (2 layers) + mean/max pooling + Focal Loss to address class imbalance
6. **Training**: early stopping + learning rate scheduling
7. **Evaluation**: F1 (macro/micro/per-label) + confusion matrices + per-label threshold tuning

## Results

| Model | F1 Macro | F1 Micro | Notes |
|---|---|---|---|
| RNN | ~0.60 (before tuning), higher after threshold tuning | ~0.73 -> ~0.74 | See `RNN/` for the full classification report |
| LSTM | _(to be filled in after the final run)_ | _(to be filled in)_ | See `LSTM/` for the full classification report |

> **Note:** The gap between macro F1 and the target score is tied to how rare some labels are in the real data (threat: ~478 examples out of ~160k, under 0.3%). Several techniques were tried to compensate (augmentation, focal loss, class balancing, threshold tuning). See the classification report and confusion matrix in each folder for the per-label breakdown.

## Project structure

```
.
├── RNN/
│   ├── RNN_training.ipynb
│   ├── results.pdf
│   └── rnn_confusion_matrices.png
├── LSTM/
│   ├── LSTM_training.ipynb
│   ├── results.pdf
│   └── lstm_confusion_matrices.png
├── requirements.txt
├── LICENSE
└── README.md
```

## How to run

1. Download the dataset (`train.csv`) from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge)
2. Download the GloVe embeddings: `glove.6B.100d.txt` from [nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
3. Install dependencies: `pip install -r requirements.txt`
4. Run the desired notebook (`RNN/RNN_training.ipynb` or `LSTM/LSTM_training.ipynb`) on an environment with a GPU (developed and tested on Google Colab)
5. (Optional) To enable LLM augmentation, set the `ANTHROPIC_API_KEY` environment variable

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) - 159,571 Wikipedia comments, each labeled across 6 toxicity categories.
