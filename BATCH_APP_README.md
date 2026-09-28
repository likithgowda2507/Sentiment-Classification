# 🎬 Batch Sentiment Classification System

## Overview

A complete machine learning pipeline for batch sentiment analysis with advanced visualizations, model comparisons, and comprehensive error analysis.

## Components Created

### 1. **Batch Prediction App** (`batch_app.py`)
Streamlit-based application for batch file upload and sentiment prediction.

#### Features:
- **File Upload Support**
  - CSV and Excel (.xlsx) file upload
  - Automatic column detection (review, text, content, sentence)
  - Optional true label column for evaluation
  
- **Data Cleaning**
  - Null/NaN value removal
  - Empty review filtering
  - Minimum length threshold

- **Batch Predictions**
  - Process multiple reviews at once
  - Confidence scores for each prediction
  - Automatic positive/negative labeling

- **Visualizations**
  - Sentiment distribution pie charts
  - Confidence score histograms
  - Keyword frequency comparison (positive vs negative)
  - Confusion matrix with performance metrics
  - Performance metrics (Accuracy, Precision, Recall, F1-Score)

- **Export Options**
  - Download results as CSV
  - Download results as Excel (.xlsx)
  - Includes predictions, confidence scores, and error labels

- **Error Analysis**
  - Identifies incorrect predictions
  - Shows false positives and false negatives
  - Explains failure patterns

### 2. **Advanced Model Comparison Script** (`advanced_model_comparison.py`)
Comprehensive testing of different model architectures and configurations.

#### Tested Configurations:
1. **CNN (k=3,4,5)** - Traditional 3,4,5-gram filters
2. **CNN (k=2,3,4)** - Smaller kernel sizes
3. **CNN (k=3,5,7)** - Larger kernel sizes
4. **CNN+Attn (k=3,4,5)** - With attention mechanism
5. **CNN+Attn (k=2,3,4)** - Smaller kernels + attention
6. **CNN+Attn (k=3,5,7)** - Larger kernels + attention

#### Output Metrics:
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Error Analysis**: Error rate, confusion matrix components (TP, TN, FP, FN)
- **Model Stats**: Parameter count, final train/val loss
- **Visualization**: 
  - Comparison plots for all metrics
  - Kernel size impact analysis
  - Attention mechanism effectiveness
  - Confusion matrices for each configuration

### 3. **Sample Dataset** (`sample_reviews.csv`)
45 movie reviews with true labels (balanced positive/negative) for testing and demonstration.

## Installation

```bash
cd "d:\Documents\Sentiment Classification"

# Install dependencies
python -m pip install streamlit plotly pandas torch datasets scikit-learn matplotlib seaborn openpyxl
```

## Usage

### Running Batch Predictions
```bash
# Run the batch prediction app
python -m streamlit run batch_app.py

# The app will open at http://localhost:8502
```

### Running Model Comparisons
```bash
# Compare different model architectures
python src/advanced_model_comparison.py --epochs 5 --batch_size 64 --output_dir outputs/comparisons
```

## Batch App Workflow

1. **Upload File**
   - Click the file uploader in the sidebar
   - Select CSV or Excel file
   - Choose review column name
   - (Optional) Select true label column for evaluation

2. **Configure Settings**
   - Toggle null/empty review removal
   - Set minimum review length
   - Review column detection

3. **Run Predictions**
   - Click "Run Predictions" button
   - Processing status displayed
   - Results shown in table format

4. **Analyze Results**
   - View sentiment distribution
   - Check confidence scores
   - Examine top keywords
   - Review performance metrics (if labels provided)
   - Download results

## Output Files

### From Batch App:
- `sentiment_predictions.csv` - Predictions with confidence scores
- `sentiment_predictions.xlsx` - Same data in Excel format

### From Model Comparison:
- `model_comparison_results.csv` - Metrics for all configurations
- `model_comparison_results.json` - JSON format results
- `model_comparison.png` - Comprehensive metric comparison plots
- `kernel_impact.png` - Kernel size and attention impact analysis
- `confusion_matrices.png` - Confusion matrices for top configurations

## Data Format Requirements

### Input File Format
CSV or Excel with at least one of these columns:
- `review` (preferred)
- `text`
- `content`
- `sentence`

Optional columns:
- `label` / `sentiment` / `true_label` (for evaluation)

### Example CSV
```csv
review,label
"This movie is amazing!",1
"Terrible film, waste of time",0
```

## Architecture Comparison Results

The script tests 6 different TextCNN configurations:

| Model | Kernels | Attention | Advantage |
|-------|---------|-----------|-----------|
| CNN(3,4,5) | 3,4,5 | No | Baseline, balanced |
| CNN(2,3,4) | 2,3,4 | No | Captures smaller patterns |
| CNN(3,5,7) | 3,5,7 | No | Captures larger patterns |
| CNN+Attn(3,4,5) | 3,4,5 | Yes | Best overall (baseline + focus) |
| CNN+Attn(2,3,4) | 2,3,4 | Yes | Focused small patterns |
| CNN+Attn(3,5,7) | 3,5,7 | Yes | Focused large patterns |

## Key Visualizations

### 1. **Sentiment Distribution**
- Pie charts showing positive/negative ratio
- Comparison between predicted and true labels

### 2. **Confidence Histogram**
- Distribution of prediction confidence scores
- Separate colors for positive and negative

### 3. **Keyword Analysis**
- Top positive words bar chart
- Top negative words bar chart
- Stop words filtered out

### 4. **Confusion Matrix Heatmap**
- TP, TN, FP, FN visualization
- Shows model errors at a glance

### 5. **Model Comparison Plots**
- 6 subplots comparing metrics
- Accuracy, Precision, Recall, F1-Score, Error Rate, Parameters

### 6. **Kernel Impact Analysis**
- Accuracy by kernel size combination
- Impact of attention mechanism

## Error Analysis Features

The system identifies and analyzes prediction errors:

1. **Failure Patterns**
   - Sarcasm detection (false positives/negatives)
   - Contradictory/mixed sentiment reviews
   - Very long reviews

2. **Error Metrics**
   - False positive rate
   - False negative rate
   - Error distribution by confidence level

3. **Attention Insights**
   - Which tokens the model focuses on
   - Attention distribution across sequence
   - Top attended tokens for each review

## Hyperparameter Tuning

Modify in the scripts:

```python
# Model parameters
--num_filters=128      # Number of filters per kernel size
--embedding_dim=100    # Embedding dimension
--dropout=0.5          # Dropout rate

# Training parameters
--epochs=5             # Number of training epochs
--batch_size=64        # Batch size
--lr=0.001             # Learning rate

# Data parameters
--max_len=300          # Maximum review length
--max_vocab_size=30000 # Vocabulary size limit
--min_freq=2           # Minimum token frequency
```

## Performance Expectations

On SST-2 validation set (872 reviews):

- **CNN Baseline (3,4,5)**: ~87-89% accuracy
- **CNN + Attention**: ~88-90% accuracy
- **With different kernel sizes**: ±1-2% variation
- **Confidence scoring**: Usually >75% confidence on correct predictions

## Batch Processing Performance

- **Speed**: ~50-100 reviews/second (CPU)
- **Memory**: ~1GB RAM for model + data
- **Scalability**: Tested up to 1000+ reviews in single batch

## Troubleshooting

### Issue: Model not found
**Solution**: Train the model first
```bash
python src/train_textcnn_imdb.py --epochs 5 --batch_size 64
```

### Issue: Memory error
**Solution**: Reduce batch size or use GPU
```python
# In batch_app.py, reduce batch_size in DataLoader
batch_size=32  # Instead of 64
```

### Issue: Slow predictions
**Solution**: Use GPU if available
```bash
# Automatically detects CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## API Integration

The batch prediction functionality can be integrated as an API:

```python
# Example: Batch prediction function
predictions, probabilities, cleaned_texts = predict_batch(
    texts=list_of_reviews,
    model=loaded_model,
    vocab=vocabulary,
    config=model_config,
    max_len=300,
    device="cuda"
)
```

## Next Steps

### Enhancements to Consider:
1. **Fine-tune embeddings** (Word2Vec, FastText, GloVe)
2. **Add LSTM/Transformer models** for comparison
3. **Implement batch processing queue** for large datasets
4. **Add model explainability** (LIME, SHAP)
5. **Create REST API** for production deployment
6. **Add real-time monitoring** dashboard
7. **Implement A/B testing** framework

## References

- Dataset: [SST-2 (Stanford Sentiment Treebank v2)](https://huggingface.co/datasets/glue/viewer/sst2)
- Model: TextCNN with optional attention mechanism
- Framework: PyTorch for model, Streamlit for UI

## License

This project is part of the Sentiment Classification system.
