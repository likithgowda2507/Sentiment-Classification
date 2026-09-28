import argparse
import html
import os
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


# ------------------------------
# Reproducibility
# ------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------
# Text preprocessing
# ------------------------------
HTML_TAG_RE = re.compile(r"<.*?>")
MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase text, strip HTML tags, punctuation, and normalize spaces."""
    text = html.unescape(text)
    text = text.lower()
    text = HTML_TAG_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer after preprocessing."""
    return text.split()


# ------------------------------
# Vocabulary and encoding
# ------------------------------
SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1}


def build_vocab(texts: List[str], max_vocab_size: int = 30000, min_freq: int = 2) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = dict(SPECIAL_TOKENS)
    for token, freq in counter.most_common():
        if len(vocab) >= max_vocab_size:
            break
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    token_ids = [vocab.get(tok, SPECIAL_TOKENS["<unk>"]) for tok in tokenize(text)]
    if len(token_ids) < max_len:
        token_ids += [SPECIAL_TOKENS["<pad>"]] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids


class IMDBTorchDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int):
        self.encoded = [encode_text(t, vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.encoded[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


# ------------------------------
# Embedding initialization
# ------------------------------
def load_glove_embeddings(vocab: Dict[str, int], embedding_dim: int = 100) -> np.ndarray:
    """Load GloVe vectors from torchtext if available; random-init OOV tokens."""
    matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
    matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)

    try:
        from torchtext.vocab import GloVe

        glove = GloVe(name="6B", dim=embedding_dim)
        hit = 0
        for token, idx in vocab.items():
            if token in glove.stoi:
                matrix[idx] = glove[token].numpy()
                hit += 1
        print(f"[INFO] Loaded GloVe embeddings. Coverage: {hit}/{len(vocab)} ({hit/len(vocab):.2%})")
    except Exception as e:
        print(f"[WARN] Could not load GloVe embeddings ({e}). Falling back to random embeddings.")
    return matrix


def load_distilbert_embeddings(vocab: Dict[str, int], embedding_dim: int = 768) -> np.ndarray:
    """Extract static embeddings from DistilBERT for vocab tokens."""
    try:
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("[WARN] transformers not installed. Falling back to random embeddings.")
        matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
        matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)
        return matrix

    try:
        print("[INFO] Loading DistilBERT tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModel.from_pretrained("distilbert-base-uncased", output_hidden_states=True)
        model.eval()

        # Initialize embedding matrix
        matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
        matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)

        # Extract embeddings for tokens in vocab
        with torch.no_grad():
            hit = 0
            for token, idx in vocab.items():
                if token in ["<pad>", "<unk>"]:
                    continue
                try:
                    # Tokenize and get embedding
                    token_ids = tokenizer.encode(token, add_special_tokens=False)
                    if len(token_ids) > 0:
                        token_ids = torch.tensor([token_ids])
                        outputs = model(token_ids, output_hidden_states=True)
                        # Use last hidden state, averaged over subword tokens
                        hidden = outputs.last_hidden_state[0].mean(dim=0).numpy()
                        if len(hidden) == embedding_dim:
                            matrix[idx] = hidden.astype(np.float32)
                            hit += 1
                except Exception:
                    pass

        print(f"[INFO] Loaded DistilBERT embeddings. Coverage: {hit}/{len(vocab)} ({hit/len(vocab):.2%})")

    except Exception as e:
        print(f"[WARN] Could not load DistilBERT embeddings ({e}). Falling back to random embeddings.")
        matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
        matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)

    return matrix


def fit_temperature(model, loader, device, init_temp: float = 1.0) -> float:
    """Simple temperature scaling on validation set to calibrate probabilities."""
    model.eval()
    temps = torch.tensor([init_temp], requires_grad=True, device=device)
    nll_criterion = nn.BCEWithLogitsLoss()

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits_list.append(logits.detach())
            labels_list.append(y.detach())

    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list)

    optimizer = torch.optim.LBFGS([temps], lr=0.1, max_iter=50)

    def eval():
        optimizer.zero_grad()
        scaled = logits_all / temps
        loss = nll_criterion(scaled, labels_all)
        loss.backward()
        return loss

    try:
        optimizer.step(eval)
    except Exception:
        pass

    temp = temps.detach().item()
    print(f"[INFO] Fitted temperature: {temp:.4f}")
    return temp


# ------------------------------
# CNN model
# ------------------------------
class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: List[int],
        dropout: float,
        pad_idx: int,
        pretrained_embeddings: np.ndarray = None,
        freeze_embeddings: bool = False,
        use_attention: bool = False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
            self.embedding.weight.requires_grad = not freeze_embeddings

        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) for k in kernel_sizes]
        )
        self.use_attention = use_attention
        if use_attention:
            self.attn_proj = nn.Linear(embedding_dim, embedding_dim)
            self.attn_vector = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)
        fc_in = num_filters * len(kernel_sizes) + (embedding_dim if use_attention else 0)
        self.fc = nn.Linear(fc_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, emb_dim]
        emb_t = emb.permute(0, 2, 1)  # [batch, emb_dim, seq_len]

        # Optional attention over token embeddings
        attn_weights = None
        if self.use_attention:
            # compute attention scores over sequence positions
            proj = torch.tanh(self.attn_proj(emb))  # [batch, seq_len, emb_dim]
            scores = self.attn_vector(proj).squeeze(-1)  # [batch, seq_len]
            attn_weights = torch.softmax(scores, dim=1)  # [batch, seq_len]
            context = (attn_weights.unsqueeze(-1) * emb).sum(dim=1)  # [batch, emb_dim]

        conv_outs = [torch.relu(conv(emb_t)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]
        features = torch.cat(pooled, dim=1)
        if self.use_attention:
            features = torch.cat([context, features], dim=1)

        features = self.dropout(features)
        logits = self.fc(features).squeeze(1)
        if attn_weights is not None:
            return logits, attn_weights
        return logits


@dataclass
class TrainHistory:
    train_loss: List[float]
    val_loss: List[float]


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = criterion(logits, y)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_predictions(model, loader, device):
    model.eval()
    probs_list, y_list = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_list.extend(probs.tolist())
        y_list.extend(y.numpy().tolist())
    y_true = np.array(y_list).astype(int)
    y_prob = np.array(probs_list)
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


def train_model(
    model,
    train_loader,
    val_loader,
    epochs,
    lr,
    device,
    early_stopping_patience: int = 3,
    use_scheduler: bool = True,
):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2) if use_scheduler else None
    history = TrainHistory(train_loss=[], val_loss=[])

    best_val = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss = evaluate_loss(model, val_loader, criterion, device)
        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        print(f"Epoch {epoch:02d}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

        if scheduler is not None:
            scheduler.step(va_loss)

        # Early stopping
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stopping_patience:
                print(f"[INFO] Early stopping triggered (patience={early_stopping_patience})")
                break

    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def plot_losses(history: TrainHistory, save_path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history.train_loss, label="Train Loss")
    plt.plot(history.val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)


def sample_predictions(texts: List[str], y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n: int = 5):
    print("\n=== Correct Predictions (sample) ===")
    correct_idx = np.where(y_true == y_pred)[0]
    for idx in correct_idx[:n]:
        label = "positive" if y_pred[idx] == 1 else "negative"
        print(f"Prob={y_prob[idx]:.3f} | Pred={label} | True={label}")
        print(texts[idx][:250].replace("\n", " "))
        print("-")


def analyze_failure_cases(texts: List[str], y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n: int = 3):
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("\nNo failure cases found on current split.")
        return

    def has_sarcasm_markers(t: str) -> bool:
        markers = ["yeah right", "as if", "sure", "totally", "great job", "what a masterpiece"]
        low = t.lower()
        return any(m in low for m in markers)

    def has_mixed_sentiment(t: str) -> bool:
        low = t.lower()
        contrast = ["but", "however", "although", "though", "yet"]
        return any(c in low.split() for c in contrast)

    long_threshold = 350

    sarcasm = [i for i in wrong_idx if has_sarcasm_markers(texts[i])]
    mixed = [i for i in wrong_idx if has_mixed_sentiment(texts[i])]
    long_reviews = [i for i in wrong_idx if len(tokenize(clean_text(texts[i]))) >= long_threshold]

    print("\n=== Failure Analysis ===")

    def print_group(indices, title):
        print(f"\n{title} (showing up to {n})")
        if not indices:
            print("No clear examples found with heuristic filters.")
            return
        for i in indices[:n]:
            pred = "positive" if y_pred[i] else "negative"
            true = "positive" if y_true[i] else "negative"
            print(f"Prob={y_prob[i]:.3f} | Pred={pred} | True={true} | length={len(tokenize(clean_text(texts[i])))}")
            print(texts[i][:300].replace("\n", " "))
            print("-")

    print_group(sarcasm, "Sarcasm-like cases")
    print_group(mixed, "Mixed sentiment cases")
    print_group(long_reviews, "Long review cases")


def visualize_top_ngrams(model: TextCNN, texts: List[str], vocab: Dict[str, int], max_len: int, device: str, save_path: str):
    """Optional: inspect high-activation n-grams from each kernel size."""
    id2tok = {idx: tok for tok, idx in vocab.items()}
    model.eval()
    records = []

    with torch.no_grad():
        for text in texts[:500]:
            enc = torch.tensor([encode_text(clean_text(text), vocab, max_len)], dtype=torch.long).to(device)
            emb = model.embedding(enc).permute(0, 2, 1)
            for conv in model.convs:
                k = conv.kernel_size[0]
                out = torch.relu(conv(emb)).squeeze(0)  # [num_filters, seq-k+1]
                max_vals, max_pos = out.max(dim=1)
                for f_idx in range(out.shape[0]):
                    pos = max_pos[f_idx].item()
                    token_ids = enc[0, pos : pos + k].tolist()
                    ngram = " ".join(id2tok.get(tid, "<unk>") for tid in token_ids if tid != SPECIAL_TOKENS["<pad>"])
                    if ngram:
                        records.append((k, max_vals[f_idx].item(), ngram))

    top = sorted(records, key=lambda x: x[1], reverse=True)[:30]
    if not top:
        return

    labels = [f"k={k}: {ngram[:35]}" for k, _, ngram in top][:15]
    values = [score for _, score, _ in top][:15]

    plt.figure(figsize=(10, 6))
    plt.barh(labels[::-1], values[::-1])
    plt.title("Top CNN-Activated n-grams (heuristic)")
    plt.xlabel("Activation")
    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close()


def train_baseline_logreg(train_texts, train_labels, test_texts, test_labels):
    print("\n=== Baseline: Logistic Regression (Bag-of-Words) ===")
    vect = CountVectorizer(max_features=30000, ngram_range=(1, 2))
    X_train = vect.fit_transform(train_texts)
    X_test = vect.transform(test_texts)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    print(f"Accuracy : {accuracy_score(test_labels, preds):.4f}")
    print(f"Precision: {precision_score(test_labels, preds):.4f}")
    print(f"Recall   : {recall_score(test_labels, preds):.4f}")
    print(f"F1-score : {f1_score(test_labels, preds):.4f}")


def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading SST-2 dataset from Hugging Face...")
    ds = load_dataset("glue", "sst2")

    train_raw = ds["train"]
    val_raw = ds["validation"]  # SST-2 test split is unlabeled in HF; use validation for evaluation.

    train_texts = [clean_text(x) for x in train_raw["sentence"]]
    train_labels = train_raw["label"]
    val_texts = [clean_text(x) for x in val_raw["sentence"]]
    val_labels = val_raw["label"]

    # Train/validation split from training set for model selection.
    val_size = int(len(train_texts) * args.val_ratio)
    indices = np.arange(len(train_texts))
    np.random.shuffle(indices)

    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]

    x_train = [train_texts[i] for i in tr_idx]
    y_train = [train_labels[i] for i in tr_idx]
    x_val = [train_texts[i] for i in val_idx]
    y_val = [train_labels[i] for i in val_idx]

    print("Building vocabulary...")
    vocab = build_vocab(x_train, max_vocab_size=args.max_vocab_size, min_freq=args.min_freq)
    print(f"Vocab size: {len(vocab)}")

    pretrained = None
    embedding_dim = args.embedding_dim
    if args.embedding_type == "glove":
        pretrained = load_glove_embeddings(vocab, embedding_dim=args.embedding_dim)
    elif args.embedding_type == "distilbert":
        embedding_dim = 768  # DistilBERT default
        pretrained = load_distilbert_embeddings(vocab, embedding_dim=768)

    train_ds = IMDBTorchDataset(x_train, y_train, vocab, args.max_len)
    val_ds = IMDBTorchDataset(x_val, y_val, vocab, args.max_len)
    test_ds = IMDBTorchDataset(val_texts, val_labels, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        num_filters=args.num_filters,
        kernel_sizes=[3, 4, 5],
        dropout=args.dropout,
        pad_idx=SPECIAL_TOKENS["<pad>"],
        pretrained_embeddings=pretrained,
        freeze_embeddings=args.freeze_embeddings,
        use_attention=args.use_attention,
    ).to(device)

    history = train_model(
        model,
        train_loader,
        val_loader,
        args.epochs,
        args.lr,
        device,
        early_stopping_patience=args.early_stopping_patience,
        use_scheduler=args.use_scheduler,
    )
    plot_losses(history, os.path.join(args.output_dir, "loss_curve.png"))

    y_true, y_pred, y_prob = evaluate_predictions(model, test_loader, device)
    print_metrics(y_true, y_pred)

    sample_predictions(val_raw["sentence"], y_true, y_pred, y_prob, n=5)
    analyze_failure_cases(val_raw["sentence"], y_true, y_pred, y_prob, n=3)

    if args.visualize_ngrams:
        if hasattr(model, "convs"):
            visualize_top_ngrams(
                model,
                val_raw["sentence"],
                vocab,
                args.max_len,
                device,
                os.path.join(args.output_dir, "top_ngrams.png"),
            )
            print("Saved top n-gram activation plot.")
        print("Saved top n-gram activation plot.")

    if args.run_baseline:
        train_baseline_logreg(x_train, y_train, val_texts, val_labels)

    # Optional temperature scaling
    temperature = None
    if args.calibrate:
        temperature = fit_temperature(model, val_loader, device)

    model_path = os.path.join(args.output_dir, "textcnn_imdb.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "config": vars(args),
            "temperature": temperature,
        },
        model_path,
    )
    print(f"Model saved to: {model_path}")
    print(f"Loss plot saved to: {os.path.join(args.output_dir, 'loss_curve.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TextCNN sentiment classifier on IMDB")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--embedding_type", type=str, choices=["random", "glove", "distilbert"], default="random")
    parser.add_argument("--freeze_embeddings", action="store_true")
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_baseline", action="store_true")
    parser.add_argument("--visualize_ngrams", action="store_true")
    parser.add_argument("--use_attention", action="store_true", help="Enable attention over tokens")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--use_scheduler", action="store_true", help="Use ReduceLROnPlateau scheduler")
    parser.add_argument("--calibrate", action="store_true", help="Fit temperature scaling after training")

    args = parser.parse_args()
    main(args)
