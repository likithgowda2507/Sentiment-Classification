import os
import re
import string
import html
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List


# ===========================
# Text preprocessing (copied from train script)
# ===========================
HTML_TAG_RE = re.compile(r"<.*?>")
MULTI_SPACE_RE = re.compile(r"\s+")
SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1}


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


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    token_ids = [vocab.get(tok, SPECIAL_TOKENS["<unk>"]) for tok in tokenize(text)]
    if len(token_ids) < max_len:
        token_ids += [SPECIAL_TOKENS["<pad>"]] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids


# ===========================
# TextCNN model (copied from train script)
# ===========================
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

        attn_weights = None
        if self.use_attention:
            proj = torch.tanh(self.attn_proj(emb))
            scores = self.attn_vector(proj).squeeze(-1)
            attn_weights = torch.softmax(scores, dim=1)
            context = (attn_weights.unsqueeze(-1) * emb).sum(dim=1)

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


# ===========================
# Visualization helpers
# ===========================
def plot_token_importance(tokens: List[str], attention_scores: np.ndarray) -> plt.Figure:
    """Plot token importance based on attention weights."""
    # Filter out padding tokens
    non_pad_idx = [i for i, t in enumerate(tokens) if t != "<pad>"]
    if not non_pad_idx:
        return None
    
    filtered_tokens = [tokens[i] for i in non_pad_idx]
    filtered_scores = [attention_scores[i] for i in non_pad_idx]
    
    # Get top 15 tokens
    top_idx = np.argsort(filtered_scores)[-15:][::-1]
    top_tokens = [filtered_tokens[i] for i in top_idx]
    top_scores = [filtered_scores[i] for i in top_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if score > np.median(filtered_scores) else '#3498db' for score in top_scores]
    ax.barh(top_tokens[::-1], top_scores[::-1], color=colors[::-1])
    ax.set_xlabel("Attention Weight", fontsize=11)
    ax.set_title("Top Influential Tokens", fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(top_scores) * 1.1)
    plt.tight_layout()
    return fig


def plot_filter_activations(model: nn.Module, x: torch.Tensor, device: str) -> plt.Figure:
    """Visualize activation scores from CNN filters."""
    with torch.no_grad():
        emb = model.embedding(x)  # [batch, seq_len, emb_dim]
        emb_t = emb.permute(0, 2, 1)  # [batch, emb_dim, seq_len]
        
        filter_activations = []
        kernel_sizes = []
        
        for i, conv in enumerate(model.convs):
            out = torch.relu(conv(emb_t))  # [batch, num_filters, seq_len-k+1]
            max_act = out.max(dim=2).values.mean(dim=0).cpu().numpy()  # [num_filters]
            kernel_size = conv.kernel_size[0]
            
            for j, act in enumerate(max_act):
                filter_activations.append(act)
                kernel_sizes.append(f"k{kernel_size}_f{j+1}")
        
        # Get top 12 filters
        top_idx = np.argsort(filter_activations)[-12:][::-1]
        top_acts = [filter_activations[i] for i in top_idx]
        top_names = [kernel_sizes[i] for i in top_idx]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_acts)))
        ax.bar(range(len(top_acts)), top_acts, color=colors)
        ax.set_xticks(range(len(top_acts)))
        ax.set_xticklabels(top_names, rotation=45, ha='right')
        ax.set_ylabel("Average Activation", fontsize=11)
        ax.set_title("Top CNN Filter Activations", fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return fig


def create_sentiment_gauge(prob: float) -> go.Figure:
    """Create a sentiment probability gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sentiment Score"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "#ff6b6b"},    # Strong negative
                {'range': [25, 50], 'color': "#ffa500"},   # Weak negative
                {'range': [50, 75], 'color': "#a3d5ff"},   # Weak positive
                {'range': [75, 100], 'color': "#2ecc71"},  # Strong positive
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_token_distribution(tokens: List[str], attention_scores: np.ndarray) -> plt.Figure:
    """Plot distribution of attention across tokens."""
    non_pad_idx = [i for i, t in enumerate(tokens) if t != "<pad>"]
    if not non_pad_idx:
        return None
    
    filtered_scores = [attention_scores[i] for i in non_pad_idx]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(filtered_scores, marker='o', linewidth=2, markersize=4, color='#3498db', alpha=0.7)
    ax.fill_between(range(len(filtered_scores)), filtered_scores, alpha=0.3, color='#3498db')
    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Attention Weight", fontsize=11)
    ax.set_title("Attention Distribution Across Sequence", fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


# ===========================
# Streamlit App
# ===========================
st.set_page_config(page_title="Sentiment Classifier", layout="centered")

st.title("🎬 Sentiment Classifier")
st.markdown("**Predict whether text has positive or negative sentiment**")

# Load model and vocab (cached)
@st.cache_resource
def load_model_and_vocab():
    checkpoint_path = "outputs/textcnn_imdb.pt"
    
    if not os.path.exists(checkpoint_path):
        st.error(f"❌ Model not found at {checkpoint_path}. Please train the model first by running:")
        st.code("py -3 src/train_textcnn_imdb.py --epochs 5 --batch_size 64 --embedding_type random --run_baseline --visualize_ngrams")
        st.stop()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = checkpoint["vocab"]
    config = checkpoint["config"]
    
    # Reconstruct model
    device = "cpu"
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=config["embedding_dim"],
        num_filters=config["num_filters"],
        kernel_sizes=[3, 4, 5],
        dropout=config["dropout"],
        pad_idx=SPECIAL_TOKENS["<pad>"],
        pretrained_embeddings=None,
        freeze_embeddings=config["freeze_embeddings"],
        use_attention=config.get("use_attention", False),
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    temperature = checkpoint.get("temperature", None)
    model.eval()
    
    return model, vocab, config, device, temperature


try:
    model, vocab, config, device, temperature = load_model_and_vocab()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Input text
text_input = st.text_area(
    "Enter text to classify:",
    placeholder="Type a movie review or any text here...",
    height=120
)

# Predict button
if st.button("🔮 Predict", use_container_width=True):
    if not text_input.strip():
        st.warning("⚠️ Please enter some text to classify.")
    else:
        # Preprocess
        cleaned = clean_text(text_input)
        encoded = encode_text(cleaned, vocab, config["max_len"])
        
        # Inference
        with torch.no_grad():
            x = torch.tensor([encoded], dtype=torch.long).to(device)
            out = model(x)
            attn_weights = None
            if isinstance(out, tuple):
                logits, attn_weights = out
            else:
                logits = out
            logits_np = logits.cpu()
            if temperature is not None:
                prob = float(torch.sigmoid(logits_np / float(temperature)).numpy()[0])
            else:
                prob = float(torch.sigmoid(logits_np).numpy()[0])
        
        pred_label = "positive" if prob >= 0.5 else "negative"
        confidence = prob if prob >= 0.5 else 1 - prob
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        
        col1, col2 = st.columns(2)
        with col1:
            if pred_label == "positive":
                st.markdown(f"### 😊 **{pred_label.upper()}**")
            else:
                st.markdown(f"### 😞 **{pred_label.upper()}**")
        
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        
        # Sentiment gauge
        st.markdown("**Probability Gauge:**")
        gauge_fig = create_sentiment_gauge(prob)
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Model Interpretation")
        
        # CNN filter activations
        st.markdown("**CNN Filter Activations:**")
        st.markdown("*Which filters are most active for this input*")
        filter_fig = plot_filter_activations(model, x, device)
        if filter_fig:
            st.pyplot(filter_fig, use_container_width=True)
        
        # Attention visualization (if available)
        if attn_weights is not None:
            st.markdown("---")
            st.markdown("**Attention Mechanism Analysis:**")
            
            id2tok = {idx: tok for tok, idx in vocab.items()}
            attn = attn_weights.cpu().numpy()[0]
            tokens = [id2tok.get(tid, "<unk>") for tid in encoded]
            
            # Tab layout for different attention visualizations
            tab1, tab2, tab3 = st.tabs(["Top Tokens", "Attention Distribution", "Full Details"])
            
            with tab1:
                st.markdown("*Top 15 most influential tokens*")
                token_fig = plot_token_importance(tokens, attn)
                if token_fig:
                    st.pyplot(token_fig, use_container_width=True)
            
            with tab2:
                st.markdown("*How attention spreads across the sequence*")
                dist_fig = plot_token_distribution(tokens, attn)
                if dist_fig:
                    st.pyplot(dist_fig, use_container_width=True)
            
            with tab3:
                st.markdown("*Detailed token-by-token attention scores (top 20)*")
                non_pad_idx = [i for i, t in enumerate(tokens) if t != "<pad>"]
                filtered_tokens = [tokens[i] for i in non_pad_idx]
                filtered_scores = [float(attn[i]) for i in non_pad_idx]
                
                top_idx = np.argsort(filtered_scores)[-20:][::-1]
                top_tokens = [filtered_tokens[i] for i in top_idx]
                top_scores = [filtered_scores[i] for i in top_idx]
                
                df_attn = {"Token": top_tokens, "Attention Score": [f"{s:.4f}" for s in top_scores]}
                st.dataframe(df_attn, use_container_width=True)
        else:
            st.info("ℹ️ Attention mechanism not enabled in this model. Using filter activations to interpret predictions.")
        
        st.markdown("---")
        
        # Show cleaned text
        with st.expander("📝 See cleaned text"):
            st.text(cleaned)


# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.markdown(
    """
    This app uses a **TextCNN** model trained on movie reviews
    to predict sentiment (positive or negative).
    
    **Model Architecture:**
    - Embedding: Random init
    - Convolutions: 3, 4, 5-gram filters
    - Pooling: Max pooling
    - Dropout: 0.5
    - Output: Binary classification (sigmoid)
    
    **Dataset:** SST-2 (Stanford Sentiment Treebank)
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed with ❤️ using PyTorch & Streamlit**")
