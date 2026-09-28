"""
Batch Sentiment Classification with Model Comparisons
Streamlit app for uploading CSV/Excel files and running batch predictions with multiple model architectures
"""

import os
import io
import re
import string
import html
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import importlib.util
from typing import Dict, List, Tuple, Optional
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, Dataset

# Import from train script
import sys
import importlib.util

src_path = os.path.join(os.path.dirname(__file__), 'src', 'train_textcnn_imdb.py')
spec = importlib.util.spec_from_file_location("train_textcnn_imdb", src_path)
train_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_module)

set_seed = train_module.set_seed
clean_text = train_module.clean_text
tokenize = train_module.tokenize
build_vocab = train_module.build_vocab
encode_text = train_module.encode_text
IMDBTorchDataset = train_module.IMDBTorchDataset
TextCNN = train_module.TextCNN
evaluate_predictions = train_module.evaluate_predictions
SPECIAL_TOKENS = train_module.SPECIAL_TOKENS
load_glove_embeddings = train_module.load_glove_embeddings
accuracy_score = train_module.accuracy_score
precision_score = train_module.precision_score
recall_score = train_module.recall_score
f1_score = train_module.f1_score
confusion_matrix = train_module.confusion_matrix


# ===========================
# Configuration
# ===========================
st.set_page_config(
    page_title="Batch Sentiment Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache model loading
@st.cache_resource
def load_trained_model():
    """Load the trained TextCNN model."""
    checkpoint_path = "outputs/textcnn_imdb.pt"
    
    if not os.path.exists(checkpoint_path):
        return None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]
    
    device = "cpu"
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=[3, 4, 5],
        dropout=config["dropout"],
        pad_idx=SPECIAL_TOKENS["<pad>"],
        pretrained_embeddings=None,
        freeze_embeddings=config.get("freeze_embeddings", False),
        use_attention=config.get("use_attention", False),
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, vocab, config


# ===========================
# Alternative Model Architectures
# ===========================
class TextCNNNoAttention(nn.Module):
    """TextCNN without attention mechanism."""
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) 
             for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)
    
    def forward(self, x):
        emb = self.embedding(x)
        emb_t = emb.permute(0, 2, 1)
        conv_outs = [torch.relu(conv(emb_t)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]
        features = torch.cat(pooled, dim=1)
        features = self.dropout(features)
        logits = self.fc(features).squeeze(1)
        return logits


class TextCNNWithAttention(nn.Module):
    """TextCNN with attention mechanism."""
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) 
             for k in kernel_sizes]
        )
        self.attn_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_vector = nn.Linear(embedding_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes) + embedding_dim, 1)
    
    def forward(self, x):
        emb = self.embedding(x)
        emb_t = emb.permute(0, 2, 1)
        
        # Attention
        proj = torch.tanh(self.attn_proj(emb))
        scores = self.attn_vector(proj).squeeze(-1)
        attn_weights = torch.softmax(scores, dim=1)
        context = (attn_weights.unsqueeze(-1) * emb).sum(dim=1)
        
        # CNN
        conv_outs = [torch.relu(conv(emb_t)) for conv in self.convs]
        pooled = [torch.max(c, dim=2).values for c in conv_outs]
        features = torch.cat(pooled, dim=1)
        features = torch.cat([context, features], dim=1)
        
        features = self.dropout(features)
        logits = self.fc(features).squeeze(1)
        return logits, attn_weights


# ===========================
# Batch Prediction Functions
# ===========================
def predict_batch(
    texts: List[str],
    model,
    vocab: Dict[str, int],
    config,
    max_len: int = 300,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Predict sentiment for a batch of texts.
    Returns: y_pred, y_prob, attention_data
    """
    cleaned_texts = [clean_text(t) for t in texts]
    encoded = [encode_text(t, vocab, max_len) for t in cleaned_texts]
    
    dataset = IMDBTorchDataset(cleaned_texts, [0] * len(cleaned_texts), vocab, max_len)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    probs_list = []
    attn_list = []
    
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            out = model(x)
            
            if isinstance(out, tuple):
                logits, attn_weights = out
                attn_list.append(attn_weights.cpu().numpy())
            else:
                logits = out
            
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.extend(probs.tolist())
    
    y_prob = np.array(probs_list)
    y_pred = (y_prob >= 0.5).astype(int)
    
    return y_pred, y_prob, cleaned_texts


def extract_important_words(texts: List[str], predictions: np.ndarray, top_n: int = 20) -> Tuple[Dict, Dict]:
    """Extract most common words in positive and negative reviews."""
    pos_words = Counter()
    neg_words = Counter()
    
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be', 'been',
                  'is', 'this', 'that', 'it', 'its', 'you', 'your', 'we', 'our'}
    
    for text, pred in zip(texts, predictions):
        tokens = tokenize(clean_text(text))
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        
        if pred == 1:
            pos_words.update(tokens)
        else:
            neg_words.update(tokens)
    
    return dict(pos_words.most_common(top_n)), dict(neg_words.most_common(top_n))


# ===========================
# Visualization Functions
# ===========================
def plot_sentiment_distribution(predictions: np.ndarray, title: str = "Sentiment Distribution") -> plt.Figure:
    """Plot sentiment distribution pie chart."""
    unique, counts = np.unique(predictions, return_counts=True)
    labels = ['Negative' if x == 0 else 'Positive' for x in unique]
    colors = ['#e74c3c', '#2ecc71']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%', 
                                        colors=colors, startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    return fig


def plot_confidence_distribution(y_prob: np.ndarray) -> plt.Figure:
    """Plot confidence score distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    pos_conf = y_prob[y_prob >= 0.5]
    neg_conf = 1 - y_prob[y_prob < 0.5]
    
    ax.hist(pos_conf, bins=30, alpha=0.6, label='Positive', color='#2ecc71', edgecolor='black')
    ax.hist(neg_conf, bins=30, alpha=0.6, label='Negative', color='#e74c3c', edgecolor='black')
    
    ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Confidence Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    return fig


def plot_keyword_comparison(pos_words: Dict, neg_words: Dict) -> plt.Figure:
    """Compare top positive and negative keywords."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Positive words
    pos_labels = list(pos_words.keys())
    pos_counts = list(pos_words.values())
    ax1.barh(pos_labels[::-1], pos_counts[::-1], color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Top Positive Words', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Negative words
    neg_labels = list(neg_words.keys())
    neg_counts = list(neg_words.values())
    ax2.barh(neg_labels[::-1], neg_counts[::-1], color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Top Negative Words', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'},
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    return fig


def plot_model_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Compare different model architectures."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        data = results_df.groupby('Model')[metric].mean()
        bars = ax.bar(data.index, data.values, color=['#3498db', '#2ecc71', '#f39c12'], alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylim([0.5, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_embedding_accuracy_comparison(results_df: pd.DataFrame) -> plt.Figure:
    """Compare embedding accuracy for trainable vs frozen."""
    fig, ax = plt.subplots(figsize=(10, 6))

    trainable = results_df[results_df["Trainable"] == "Yes"].set_index("Embedding Type")
    frozen = results_df[results_df["Trainable"] == "No"].set_index("Embedding Type")

    embedding_types = sorted(set(results_df["Embedding Type"].tolist()))
    trainable_acc = trainable.reindex(embedding_types)["Accuracy"].values
    frozen_acc = frozen.reindex(embedding_types)["Accuracy"].values

    x = np.arange(len(embedding_types))
    width = 0.35

    ax.bar(x - width / 2, trainable_acc, width, label="Trainable", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, frozen_acc, width, label="Frozen", color="#e67e22", alpha=0.8)

    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Embedding Accuracy: Trainable vs Frozen", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(embedding_types, rotation=30, ha="right")
    ax.set_ylim([0.5, 1.0])
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_embedding_accuracy_heatmap(results_df: pd.DataFrame) -> plt.Figure:
    """Heatmap of embedding accuracy by trainability."""
    pivot_data = results_df.pivot(index="Embedding Type", columns="Trainable", values="Accuracy")
    pivot_data = pivot_data[["No", "Yes"]]
    pivot_data.columns = ["Frozen", "Trainable"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        cbar_kws={"label": "Accuracy"},
        ax=ax
    )
    ax.set_title("Accuracy Heatmap: Embedding vs Trainability", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tight_layout()
    return fig


def create_attention_heatmap(attn_weights: np.ndarray, tokens: List[str], title: str = "Attention Scores") -> plt.Figure:
    """Create attention heatmap for a single review."""
    # Reshape for heatmap
    non_pad_idx = [i for i, t in enumerate(tokens) if t != "<pad>"]
    if not non_pad_idx:
        return None
    
    attn_subset = attn_weights[non_pad_idx]
    tokens_subset = [tokens[i] for i in non_pad_idx]
    
    fig, ax = plt.subplots(figsize=(12, 3))
    sns.heatmap(attn_subset.reshape(1, -1), annot=np.round(attn_subset, 3), fmt='',
                cmap='YlOrRd', cbar_kws={'label': 'Attention Weight'},
                xticklabels=tokens_subset[:min(20, len(tokens_subset))], ax=ax)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


# ===========================
# Streamlit UI
# ===========================
st.title("🎬 Batch Sentiment Classification")
st.markdown("**Upload CSV/Excel files and get predictions with detailed analysis**")

# Sidebar
st.sidebar.title("⚙️ Settings")
model, vocab, config = load_trained_model()

if model is None:
    st.error("❌ Model not found. Please train the model first.")
    st.stop()

# File upload
st.sidebar.markdown("---")
st.sidebar.subheader("📁 File Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"],
    help="File must contain a 'review', 'text', 'content', 'sentence', or 'movie_review' column"
)

if uploaded_file is None:
    st.info("👈 Upload a file to get started")
    st.stop()

# Read file
try:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    st.sidebar.success(f"✅ Loaded {len(df)} reviews")
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

# Find review column
review_cols = [
    col
    for col in df.columns
    if col.lower() in ["review", "text", "content", "sentence", "movie_review"]
]
if not review_cols:
    st.error(
        "❌ No 'review', 'text', 'content', 'sentence', or 'movie_review' column found in file"
    )
    st.stop()

review_column = st.sidebar.selectbox("Select review column:", review_cols)
label_column = None

if any(col.lower() in ['label', 'sentiment', 'true_label'] for col in df.columns):
    label_cols = [col for col in df.columns if col.lower() in ['label', 'sentiment', 'true_label']]
    label_column = st.sidebar.selectbox("Select label column (optional):", [None] + label_cols)

# Data cleaning options
st.sidebar.markdown("---")
st.sidebar.subheader("🧹 Data Cleaning")
remove_null = st.sidebar.checkbox("Remove null reviews", value=True)
remove_empty = st.sidebar.checkbox("Remove empty reviews", value=True)
min_length = st.sidebar.slider("Minimum review length (chars)", 0, 100, 1)

# Process data
reviews = df[review_column].astype(str).tolist()
true_labels = None

if label_column and label_column in df.columns:
    true_labels = df[label_column].astype(int).tolist()

# Clean data
if remove_null:
    mask = [r.lower() != 'nan' and r.lower() != 'none' for r in reviews]
    reviews = [r for r, m in zip(reviews, mask) if m]
    if true_labels:
        true_labels = [l for l, m in zip(true_labels, mask) if m]

if remove_empty:
    mask = [len(r.strip()) > min_length for r in reviews]
    reviews = [r for r, m in zip(reviews, mask) if m]
    if true_labels:
        true_labels = [l for l, m in zip(true_labels, mask) if m]

st.sidebar.info(f"📊 Processing {len(reviews)} reviews")

# Run predictions
st.markdown("---")
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🔮 Predictions")

with col2:
    if st.button("Run Predictions", use_container_width=True):
        st.session_state.predictions_done = True

if "predictions_done" not in st.session_state:
    st.info("Click 'Run Predictions' to start")
    st.stop()

# Perform predictions
device = "cuda" if torch.cuda.is_available() else "cpu"

with st.spinner("🔄 Processing predictions..."):
    y_pred, y_prob, cleaned_texts = predict_batch(
        reviews, model, vocab, config,
        max_len=config.get("max_len", 300),
        device=device
    )

# Display results table
st.markdown("### Prediction Results")
total_rows = len(reviews)
default_rows = min(50, total_rows)
show_rows = st.number_input(
    "Rows to display",
    min_value=1,
    max_value=total_rows,
    value=default_rows,
    step=1
)
results_df = pd.DataFrame({
    "Review": reviews[:show_rows],
    "Predicted": ["Positive" if p == 1 else "Negative" for p in y_pred[:show_rows]],
    "Confidence": [
        f"{prob:.1%}" if p == 1 else f"{1-prob:.1%}"
        for p, prob in zip(y_pred[:show_rows], y_prob[:show_rows])
    ]
})

if true_labels:
    results_df["True Label"] = [
        "Positive" if l == 1 else "Negative" for l in true_labels[:show_rows]
    ]
    results_df["✓/✗"] = [
        "✓" if p == l else "✗"
        for p, l in zip(y_pred[:show_rows], true_labels[:show_rows])
    ]

st.dataframe(results_df, use_container_width=True)

# Statistics
st.markdown("### 📊 Statistics")
col1, col2, col3, col4 = st.columns(4)

pos_count = np.sum(y_pred == 1)
neg_count = np.sum(y_pred == 0)
avg_conf = np.mean(np.maximum(y_prob, 1 - y_prob))

with col1:
    st.metric("Positive Reviews", pos_count)
with col2:
    st.metric("Negative Reviews", neg_count)
with col3:
    st.metric("Avg Confidence", f"{avg_conf:.1%}")
with col4:
    if true_labels:
        acc = accuracy_score(true_labels, y_pred)
        st.metric("Accuracy", f"{acc:.1%}")

# Visualizations
st.markdown("---")
st.subheader("📈 Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Confidence", "Keywords", "Analysis"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = plot_sentiment_distribution(y_pred, "Predicted Sentiment Distribution")
        st.pyplot(fig)
    with col2:
        if true_labels:
            fig = plot_sentiment_distribution(np.array(true_labels), "True Sentiment Distribution")
            st.pyplot(fig)

with tab2:
    fig = plot_confidence_distribution(y_prob)
    st.pyplot(fig)

with tab3:
    pos_words, neg_words = extract_important_words(cleaned_texts, y_pred)
    if pos_words or neg_words:
        fig = plot_keyword_comparison(pos_words, neg_words)
        st.pyplot(fig)

with tab4:
    st.markdown("#### Embedding Comparison")
    results_path = os.path.join("outputs", "embedding_comparison_results.csv")
    if os.path.exists(results_path):
        embedding_results = pd.read_csv(results_path)
        if {
            "Embedding Type",
            "Trainable",
            "Accuracy",
        }.issubset(embedding_results.columns):
            fig = plot_embedding_accuracy_comparison(embedding_results)
            st.pyplot(fig)
            fig = plot_embedding_accuracy_heatmap(embedding_results)
            st.pyplot(fig)
            st.dataframe(embedding_results, use_container_width=True)
        else:
            st.warning("Embedding results file is missing required columns.")
    else:
        st.info(
            "Run the embedding comparison script to generate results. "
            "Expected file: outputs/embedding_comparison_results.csv"
        )

# Export results
st.markdown("---")
st.subheader("💾 Export Results")

export_df = pd.DataFrame({
    "Review": reviews,
    "Predicted": ["Positive" if p == 1 else "Negative" for p in y_pred],
    "Confidence": np.maximum(y_prob, 1 - y_prob),
})

if true_labels:
    export_df["True Label"] = ["Positive" if l == 1 else "Negative" for l in true_labels]
    export_df["Correct"] = [p == l for p, l in zip(y_pred, true_labels)]

# CSV export
csv = export_df.to_csv(index=False)
st.download_button(
    "📥 Download as CSV",
    csv,
    "sentiment_predictions.csv",
    "text/csv",
    use_container_width=True
)

# Excel export
illegal_characters_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")

def sanitize_excel_value(value):
    if isinstance(value, str):
        return illegal_characters_re.sub("", value)
    return value

def sanitize_excel_df(df):
    clean_df = df.copy()
    for col in clean_df.columns:
        if clean_df[col].dtype == object:
            clean_df[col] = clean_df[col].map(sanitize_excel_value)
    return clean_df

excel_buffer = io.BytesIO()
export_df_excel = sanitize_excel_df(export_df)
with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
    export_df_excel.to_excel(writer, sheet_name='Predictions', index=False)
excel_buffer.seek(0)
st.download_button(
    "📥 Download as Excel",
    excel_buffer,
    "sentiment_predictions.xlsx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)

st.success("✅ All done!")
