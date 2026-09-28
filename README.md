# CNN Sentiment Classification on IMDB (PyTorch)

This project trains a **TextCNN** model to classify movie reviews as **positive** or **negative**.

## What this includes
- Data preprocessing (lowercase, HTML removal, punctuation removal)
- Tokenization, vocabulary building, sequence padding
- TextCNN with multiple kernel sizes (3, 4, 5)
- Trainable random embeddings or optional GloVe embeddings
- Training with `BCEWithLogitsLoss` + Adam
- Validation split and loss curve visualization
- Test evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix
- Success/failure analysis with sample reviews
- Bonus baseline: Logistic Regression (Bag-of-Words)
- Optional n-gram activation visualization from CNN filters

## Project structure
- `src/train_textcnn_imdb.py`: complete runnable pipeline
- `outputs/`: saved model and plots

## Setup
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets scikit-learn matplotlib numpy
# optional for GloVe:
pip install torchtext
```

If you are using CPU only, install CPU PyTorch wheels from the official PyTorch site.

## Run (default: random embeddings)
```bash
python src/train_textcnn_imdb.py --epochs 5 --batch_size 64 --embedding_type random --run_baseline --visualize_ngrams
```

## Run with GloVe embeddings
```bash
python src/train_textcnn_imdb.py --epochs 5 --embedding_type glove --embedding_dim 100 --visualize_ngrams
```

## Key arguments
- `--max_len 300`: truncates/pads all reviews to length 300
- `--max_vocab_size 30000`: vocabulary cap
- `--dropout 0.5`: regularization
- `--run_baseline`: compares with Logistic Regression
- `--visualize_ngrams`: saves CNN-activated n-grams plot

## Outputs
In `outputs/`:
- `textcnn_imdb.pt`: trained model checkpoint + vocab + config
- `loss_curve.png`: training vs validation loss
- `top_ngrams.png` (if enabled)

## Notes on failure analysis
The script highlights mistakes and surfaces likely categories:
- **sarcasm-like** (heuristic markers)
- **mixed sentiment** (contrast words like "but", "however")
- **long reviews** (>= 350 tokens)

This is useful for diagnosing where CNNs struggle:
- Long-range dependencies in very long text
- Contradictory sentiment within one review
- Subtle tone/sarcasm not captured by local n-grams

## Typical performance expectation
TextCNN on IMDB usually reaches strong performance (often around high-80s to low-90s accuracy depending on hyperparameters and preprocessing).
