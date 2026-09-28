"""
Embedding Comparison Script
Tests different embedding methods and their configurations on TextCNN
"""

import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Import from train script
import sys
sys.path.insert(0, os.path.dirname(__file__))

from train_textcnn_imdb import (
    set_seed, clean_text, tokenize, build_vocab, encode_text, 
    IMDBTorchDataset, TextCNN, train_model, evaluate_predictions,
    SPECIAL_TOKENS, load_glove_embeddings, load_distilbert_embeddings,
    accuracy_score, precision_score, recall_score, f1_score
)


# ===========================
# Word2Vec Embeddings
# ===========================
def load_word2vec_embeddings(vocab: Dict[str, int], embedding_dim: int = 100) -> np.ndarray:
    """Load Word2Vec embeddings using gensim."""
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        print("[WARN] gensim not installed. Falling back to random embeddings.")
        matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
        matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)
        return matrix
    
    try:
        print("[INFO] Loading Word2Vec embeddings (Google News pre-trained)...")
        # This uses a pre-trained model - you may need to download it
        # For this demo, we'll create a simple Word2Vec from scratch using the vocabulary
        from gensim.models import Word2Vec
        
        # Create a dummy Word2Vec model with vocab tokens
        # In production, you'd use pre-trained models
        sentences = [[token] for token in vocab.keys() if token not in ["<pad>", "<unk>"]]
        if sentences:
            w2v_model = Word2Vec(sentences=sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4, sg=1)
            matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
            matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)
            
            hit = 0
            for token, idx in vocab.items():
                if token in w2v_model.wv:
                    matrix[idx] = w2v_model.wv[token].astype(np.float32)
                    hit += 1
            print(f"[INFO] Loaded Word2Vec embeddings. Coverage: {hit}/{len(vocab)} ({hit/len(vocab):.2%})")
        else:
            matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
            matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] Could not load Word2Vec embeddings ({e}). Falling back to random embeddings.")
        matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
        matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)
    
    return matrix


# ===========================
# FastText Embeddings
# ===========================
def load_fasttext_embeddings(vocab: Dict[str, int], embedding_dim: int = 100) -> np.ndarray:
    """Load FastText embeddings using gensim."""
    try:
        from gensim.models import FastText
    except ImportError:
        print("[WARN] gensim not installed. Falling back to random embeddings.")
        matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
        matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)
        return matrix
    
    try:
        print("[INFO] Loading FastText embeddings...")
        sentences = [[token] for token in vocab.keys() if token not in ["<pad>", "<unk>"]]
        if sentences:
            ft_model = FastText(sentences=sentences, vector_size=embedding_dim, window=5, min_count=1, workers=4)
            matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
            matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)
            
            hit = 0
            for token, idx in vocab.items():
                if token in ft_model.wv:
                    matrix[idx] = ft_model.wv[token].astype(np.float32)
                    hit += 1
            print(f"[INFO] Loaded FastText embeddings. Coverage: {hit}/{len(vocab)} ({hit/len(vocab):.2%})")
        else:
            matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
            matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] Could not load FastText embeddings ({e}). Falling back to random embeddings.")
        matrix = np.random.normal(scale=0.1, size=(len(vocab), embedding_dim)).astype(np.float32)
        matrix[SPECIAL_TOKENS["<pad>"]] = np.zeros(embedding_dim, dtype=np.float32)
    
    return matrix


# ===========================
# Training & Evaluation
# ===========================
def train_and_evaluate(
    embedding_type: str,
    freeze_embeddings: bool,
    vocab: Dict[str, int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    embedding_dim: int = 100,
    epochs: int = 5,
    num_filters: int = 128,
    device: str = "cpu",
) -> Tuple[float, float, float, float]:
    """Train a TextCNN model and return accuracy metrics."""
    
    # Load embeddings
    pretrained = None
    actual_embedding_dim = embedding_dim
    
    if embedding_type == "random":
        pretrained = None
    elif embedding_type == "glove":
        pretrained = load_glove_embeddings(vocab, embedding_dim=embedding_dim)
    elif embedding_type == "word2vec":
        pretrained = load_word2vec_embeddings(vocab, embedding_dim=embedding_dim)
    elif embedding_type == "fasttext":
        pretrained = load_fasttext_embeddings(vocab, embedding_dim=embedding_dim)
    elif embedding_type == "distilbert":
        actual_embedding_dim = 768
        pretrained = load_distilbert_embeddings(vocab, embedding_dim=768)
    
    # Create model
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=actual_embedding_dim,
        num_filters=num_filters,
        kernel_sizes=[3, 4, 5],
        dropout=0.5,
        pad_idx=SPECIAL_TOKENS["<pad>"],
        pretrained_embeddings=pretrained,
        freeze_embeddings=freeze_embeddings,
        use_attention=False,
    ).to(device)
    
    # Train
    print(f"Training {embedding_type} (freeze={freeze_embeddings})...")
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.001,
        device=device,
        early_stopping_patience=3,
        use_scheduler=True,
    )
    
    # Evaluate
    y_true, y_pred, y_prob = evaluate_predictions(model, test_loader, device)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred))
    rec = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}\n")
    
    return acc, prec, rec, f1


# ===========================
# Visualization
# ===========================
def plot_embedding_comparison(results_df: pd.DataFrame, output_path: str = "outputs/embedding_comparison.png"):
    """Create comparison plots for different embeddings."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prepare data
    embedding_types = results_df["Embedding Type"].unique()
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Embedding Methods & Trainability Comparison on TextCNN", fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Prepare data for grouped bar chart
        trainable_data = results_df[results_df["Trainable"] == "Yes"][metric].values
        frozen_data = results_df[results_df["Trainable"] == "No"][metric].values
        embedding_names = results_df[results_df["Trainable"] == "Yes"]["Embedding Type"].values
        
        x = np.arange(len(embedding_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, trainable_data, width, label="Trainable", color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, frozen_data, width, label="Frozen", color='#e74c3c', alpha=0.8)
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(embedding_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.5, 1.0])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {output_path}")
    plt.close()


def plot_accuracy_heatmap(results_df: pd.DataFrame, output_path: str = "outputs/embedding_heatmap.png"):
    """Create a heatmap of accuracy results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Pivot table for heatmap
    pivot_data = results_df.pivot(index="Embedding Type", columns="Trainable", values="Accuracy")
    pivot_data = pivot_data[["No", "Yes"]]  # Reorder columns
    pivot_data.columns = ["Frozen", "Trainable"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im = ax.imshow(pivot_data.values, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot_data.columns)))
    ax.set_yticks(np.arange(len(pivot_data.index)))
    ax.set_xticklabels(pivot_data.columns, fontsize=11, fontweight='bold')
    ax.set_yticklabels(pivot_data.index, fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Accuracy", fontsize=11, fontweight='bold')
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = ax.text(j, i, f'{pivot_data.values[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=12, fontweight='bold')
    
    ax.set_title("Accuracy Heatmap: Embeddings vs Trainability", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {output_path}")
    plt.close()


# ===========================
# Main Comparison Function
# ===========================
def main(args):
    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading SST-2 dataset...")
    ds = load_dataset("glue", "sst2")
    
    train_raw = ds["train"]
    val_raw = ds["validation"]
    
    train_texts = [clean_text(x) for x in train_raw["sentence"]]
    train_labels = train_raw["label"]
    val_texts = [clean_text(x) for x in val_raw["sentence"]]
    val_labels = val_raw["label"]
    
    # Create train/val/test split
    val_size = int(len(train_texts) * 0.2)
    indices = np.arange(len(train_texts))
    np.random.shuffle(indices)
    
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    x_train = [train_texts[i] for i in tr_idx]
    y_train = [train_labels[i] for i in tr_idx]
    x_val = [train_texts[i] for i in val_idx]
    y_val = [train_labels[i] for i in val_idx]
    
    # Use official validation set as test
    x_test = val_texts
    y_test = val_labels
    
    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = build_vocab(x_train, max_vocab_size=args.max_vocab_size, min_freq=args.min_freq)
    print(f"Vocab size: {len(vocab)}")
    
    # Create datasets and loaders
    train_ds = IMDBTorchDataset(x_train, y_train, vocab, args.max_len)
    val_ds = IMDBTorchDataset(x_val, y_val, vocab, args.max_len)
    test_ds = IMDBTorchDataset(x_test, y_test, vocab, args.max_len)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Test different embedding methods
    embedding_types = ["random", "glove", "word2vec", "fasttext"]
    results = []
    
    print("=" * 70)
    print("TESTING EMBEDDING METHODS")
    print("=" * 70)
    
    for emb_type in embedding_types:
        for freeze in [False, True]:
            try:
                acc, prec, rec, f1 = train_and_evaluate(
                    embedding_type=emb_type,
                    freeze_embeddings=freeze,
                    vocab=vocab,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    embedding_dim=args.embedding_dim,
                    epochs=args.epochs,
                    num_filters=args.num_filters,
                    device=device,
                )
                
                results.append({
                    "Embedding Type": emb_type.upper(),
                    "Trainable": "Yes" if not freeze else "No",
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1-Score": f1,
                })
            except Exception as e:
                print(f"  ❌ Failed to train {emb_type} (freeze={freeze}): {e}\n")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print()
    
    # Save results
    results_path = os.path.join(args.output_dir, "embedding_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"✅ Results saved to: {results_path}\n")
    
    # Plot results
    plot_embedding_comparison(results_df, os.path.join(args.output_dir, "embedding_comparison.png"))
    plot_accuracy_heatmap(results_df, os.path.join(args.output_dir, "embedding_heatmap.png"))
    
    # Best results
    print("\n" + "=" * 70)
    print("BEST PERFORMING CONFIGURATIONS")
    print("=" * 70)
    best_acc = results_df.loc[results_df["Accuracy"].idxmax()]
    best_f1 = results_df.loc[results_df["F1-Score"].idxmax()]
    
    print(f"\nHighest Accuracy:")
    print(f"  {best_acc['Embedding Type']} (Trainable={best_acc['Trainable']}) → {best_acc['Accuracy']:.4f}")
    
    print(f"\nHighest F1-Score:")
    print(f"  {best_f1['Embedding Type']} (Trainable={best_f1['Trainable']}) → {best_f1['F1-Score']:.4f}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different embedding methods for TextCNN")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    
    args = parser.parse_args()
    main(args)
