"""
Advanced Model Comparison Script
Compares different CNN architectures, kernel sizes, and training configurations
Generates comprehensive reports and visualizations
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import defaultdict
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import json

# Import from train script
import sys
sys.path.insert(0, os.path.dirname(__file__))

from train_textcnn_imdb import (
    set_seed, clean_text, tokenize, build_vocab, encode_text,
    IMDBTorchDataset, train_model, evaluate_predictions,
    SPECIAL_TOKENS, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


# ===========================
# Model Variants
# ===========================
class TextCNNVariant(nn.Module):
    """Base CNN model with configurable architecture."""
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: List[int],
        dropout: float,
        pad_idx: int,
        use_attention: bool = False,
    ):
        super().__init__()
        self.use_attention = use_attention
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) 
             for k in kernel_sizes]
        )
        
        if use_attention:
            self.attn_proj = nn.Linear(embedding_dim, embedding_dim)
            self.attn_vector = nn.Linear(embedding_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        fc_in = num_filters * len(kernel_sizes)
        if use_attention:
            fc_in += embedding_dim
        self.fc = nn.Linear(fc_in, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        emb_t = emb.permute(0, 2, 1)
        
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
# Training & Evaluation
# ===========================
def train_and_evaluate_model(
    model_name: str,
    kernel_sizes: List[int],
    use_attention: bool,
    vocab: Dict[str, int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 3,
    num_filters: int = 128,
    embedding_dim: int = 100,
    device: str = "cpu",
) -> Dict:
    """Train a model variant and return metrics."""
    
    model = TextCNNVariant(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        dropout=0.5,
        pad_idx=SPECIAL_TOKENS["<pad>"],
        use_attention=use_attention,
    ).to(device)
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Kernels: {kernel_sizes} | Attention: {use_attention}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}")
    
    history = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        lr=0.001,
        device=device,
        early_stopping_patience=2,
        use_scheduler=True,
    )
    
    # Evaluate
    y_true, y_pred, y_prob = evaluate_predictions(model, test_loader, device)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Error analysis
    errors = np.where(y_true != y_pred)[0]
    error_rate = len(errors) / len(y_true)
    
    results = {
        "Model": model_name,
        "Kernels": str(kernel_sizes),
        "Attention": use_attention,
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "Error Rate": float(error_rate),
        "TP": int(cm[1, 1]),
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "Parameters": sum(p.numel() for p in model.parameters()),
        "Train Loss": history.train_loss[-1] if history.train_loss else None,
        "Val Loss": history.val_loss[-1] if history.val_loss else None,
    }
    
    print(f"\nResults:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return results


# ===========================
# Visualization
# ===========================
def plot_model_comparison(results_df: pd.DataFrame, output_path: str):
    """Create comprehensive model comparison plots."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Model Architecture Comparison", fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Error Rate', 'Parameters']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        if metric == 'Parameters':
            data = results_df[['Model', metric]].drop_duplicates()
            x_pos = np.arange(len(data))
            ax.bar(x_pos, data[metric].values, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(data['Model'].values, rotation=45, ha='right')
            ax.set_ylabel('Parameter Count', fontsize=10, fontweight='bold')
            ax.set_yscale('log')
        else:
            data = results_df.groupby('Model')[metric].mean()
            x_pos = np.arange(len(data))
            colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
            ax.bar(x_pos, data.values, color=colors[:len(data)], alpha=0.7, edgecolor='black')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(data.index, rotation=45, ha='right')
            ax.set_ylabel(metric, fontsize=10, fontweight='bold')
            if metric != 'Error Rate':
                ax.set_ylim([0.5, 1.0])
        
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved to: {output_path}")
    plt.close()


def plot_kernel_size_impact(results_df: pd.DataFrame, output_path: str):
    """Analyze impact of kernel sizes on performance."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Kernel Size Impact on Performance", fontsize=14, fontweight='bold')
    
    # Accuracy by kernel size
    kernel_acc = results_df.groupby('Kernels')['Accuracy'].mean().sort_values(ascending=False)
    ax1.barh(kernel_acc.index, kernel_acc.values, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Accuracy by Kernel Sizes', fontsize=12, fontweight='bold')
    ax1.set_xlim([0.5, 1.0])
    ax1.grid(axis='x', alpha=0.3)
    
    # F1-Score by attention
    attn_f1 = results_df.groupby('Attention')['F1-Score'].mean()
    colors = ['#e74c3c', '#2ecc71']
    ax2.bar(['No Attention', 'Attention'], attn_f1.values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax2.set_title('Impact of Attention Mechanism', fontsize=12, fontweight='bold')
    ax2.set_ylim([0.5, 1.0])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f"✅ Kernel impact plot saved to: {output_path}")
    plt.close()


def plot_confusion_matrices(results_dict: Dict, output_path: str, test_data_size: int = 872):
    """Plot confusion matrices for different models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    models = list(results_dict.keys())[:6]  # Limit to 6 for visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Confusion Matrices - Different Model Configurations", fontsize=14, fontweight='bold')
    
    for idx, (ax, model_name) in enumerate(zip(axes.flat, models)):
        result = results_dict[model_name]
        cm = np.array([[result['TN'], result['FP']], [result['FN'], result['TP']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'})
        ax.set_ylabel('True', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        acc = (result['TP'] + result['TN']) / test_data_size
        ax.set_title(f"{model_name}\nAcc: {acc:.3f}", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=140, bbox_inches='tight')
    print(f"✅ Confusion matrices saved to: {output_path}")
    plt.close()


# ===========================
# Main Comparison
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
    
    # Create splits
    val_size = int(len(train_texts) * 0.2)
    indices = np.arange(len(train_texts))
    np.random.shuffle(indices)
    
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    x_train = [train_texts[i] for i in tr_idx]
    y_train = [train_labels[i] for i in tr_idx]
    x_val = [train_texts[i] for i in val_idx]
    y_val = [train_labels[i] for i in val_idx]
    
    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(val_texts)}")
    
    # Build vocabulary
    print("\nBuilding vocabulary...")
    vocab = build_vocab(x_train, max_vocab_size=args.max_vocab_size, min_freq=args.min_freq)
    print(f"Vocab size: {len(vocab)}")
    
    # Create datasets
    train_ds = IMDBTorchDataset(x_train, y_train, vocab, args.max_len)
    val_ds = IMDBTorchDataset(x_val, y_val, vocab, args.max_len)
    test_ds = IMDBTorchDataset(val_texts, val_labels, vocab, args.max_len)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Define model configurations
    configs = [
        ("CNN (k=3,4,5)", [3, 4, 5], False),
        ("CNN (k=2,3,4)", [2, 3, 4], False),
        ("CNN (k=3,5,7)", [3, 5, 7], False),
        ("CNN+Attn (k=3,4,5)", [3, 4, 5], True),
        ("CNN+Attn (k=2,3,4)", [2, 3, 4], True),
        ("CNN+Attn (k=3,5,7)", [3, 5, 7], True),
    ]
    
    # Train all models
    results_list = []
    results_dict = {}
    
    print("="*70)
    print("TRAINING DIFFERENT MODEL CONFIGURATIONS")
    print("="*70)
    
    for model_name, kernel_sizes, use_attention in configs:
        try:
            result = train_and_evaluate_model(
                model_name=model_name,
                kernel_sizes=kernel_sizes,
                use_attention=use_attention,
                vocab=vocab,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                epochs=args.epochs,
                num_filters=args.num_filters,
                embedding_dim=args.embedding_dim,
                device=device,
            )
            results_list.append(result)
            results_dict[model_name] = result
        except Exception as e:
            print(f"❌ Failed to train {model_name}: {e}\n")
    
    # Create results dataframe
    results_df = pd.DataFrame(results_list)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(results_df[['Model', 'Kernels', 'Attention', 'Accuracy', 'F1-Score', 'Error Rate']].to_string(index=False))
    
    # Save results
    results_path = os.path.join(args.output_dir, "model_comparison_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    
    # Save JSON for easy parsing
    json_path = os.path.join(args.output_dir, "model_comparison_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"✅ Results saved to: {json_path}")
    
    # Generate visualizations
    plot_model_comparison(results_df, os.path.join(args.output_dir, "model_comparison.png"))
    plot_kernel_size_impact(results_df, os.path.join(args.output_dir, "kernel_impact.png"))
    plot_confusion_matrices(results_dict, os.path.join(args.output_dir, "confusion_matrices.png"))
    
    # Best models
    print("\n" + "="*70)
    print("TOP PERFORMING CONFIGURATIONS")
    print("="*70)
    
    top_3 = results_df.nlargest(3, 'F1-Score')[['Model', 'Kernels', 'Attention', 'Accuracy', 'F1-Score']]
    print(top_3.to_string(index=False))
    
    print("\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare different TextCNN architectures")
    parser.add_argument("--output_dir", type=str, default="outputs/comparisons")
    parser.add_argument("--max_vocab_size", type=int, default=30000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=300)
    parser.add_argument("--embedding_dim", type=int, default=100)
    parser.add_argument("--num_filters", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    
    args = parser.parse_args()
    main(args)
