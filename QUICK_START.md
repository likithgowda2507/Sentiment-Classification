# Quick Start Guide - Batch Sentiment Classification

## 🚀 Getting Started in 5 Minutes

### Step 1: Train the Model (if not already done)
```bash
cd "d:\Documents\Sentiment Classification"

python src/train_textcnn_imdb.py --epochs 5 --batch_size 64 --embedding_type random --run_baseline --visualize_ngrams
```

**Expected Output:**
- `outputs/textcnn_imdb.pt` - Trained model
- `outputs/loss_curve.png` - Training visualization
- ~87-89% accuracy on test set

### Step 2: Launch the Batch Prediction App
```bash
python -m streamlit run batch_app.py
```

**The app will open at:** `http://localhost:8502`

### Step 3: Upload Your File

1. Click "Upload CSV or Excel file" in the sidebar
2. Select a file with a column containing reviews
3. Choose the review column name
4. (Optional) Select a label column if you have true sentiments

**Supported Column Names:**
- `review`, `text`, `content`, `sentence`

### Step 4: Configure & Run

1. Set data cleaning options:
   - Remove null reviews ✓
   - Remove empty reviews ✓
   - Minimum length: 1 character

2. Click "Run Predictions" button

### Step 5: Analyze Results

**Available Tabs:**
- **Distribution** - Sentiment pie charts
- **Confidence** - Prediction confidence histogram
- **Keywords** - Top positive/negative words
- **Analysis** - Confusion matrix & metrics (if labels provided)

### Step 6: Download Results

Choose output format:
- **CSV** - Universal spreadsheet format
- **Excel** - Full Excel workbook (.xlsx)

---

## 📊 Sample Data

A sample file is included: `sample_reviews.csv`

**Use it for testing:**
1. Upload `sample_reviews.csv` to the batch app
2. Select "review" as review column
3. Select "label" as label column
4. Run predictions
5. See performance metrics and visualizations

---

## 🔬 Run Model Comparison

Compare different CNN architectures:

```bash
python src/advanced_model_comparison.py \
    --epochs 3 \
    --batch_size 64 \
    --output_dir outputs/comparisons
```

**Output Files:**
- `model_comparison_results.csv` - Raw metrics
- `model_comparison.png` - Comparison charts
- `kernel_impact.png` - Architecture impact analysis
- `confusion_matrices.png` - Error visualization

**Tested Models:**
```
1. CNN (k=3,4,5)        - Baseline
2. CNN (k=2,3,4)        - Smaller patterns
3. CNN (k=3,5,7)        - Larger patterns
4. CNN+Attn (k=3,4,5)   - Baseline + attention
5. CNN+Attn (k=2,3,4)   - Small patterns + attention
6. CNN+Attn (k=3,5,7)   - Large patterns + attention
```

---

## 📁 File Structure

```
Sentiment Classification/
├── app.py                          # Single review UI
├── batch_app.py                    # Batch prediction app ← START HERE
├── sample_reviews.csv              # Test data (45 reviews)
├── BATCH_APP_README.md             # Full documentation
├── QUICK_START.md                  # This file
├── src/
│   ├── train_textcnn_imdb.py      # Training script
│   ├── advanced_model_comparison.py # Model comparison
│   └── compare_embeddings.py       # Embedding comparison
├── outputs/
│   ├── textcnn_imdb.pt            # Trained model
│   ├── loss_curve.png             # Training history
│   ├── sentiment_predictions.csv   # Generated predictions
│   └── comparisons/
│       ├── model_comparison_results.csv
│       ├── model_comparison.png
│       └── confusion_matrices.png
└── README.md
```

---

## 🎯 Key Features

### ✅ Batch Processing
- Upload CSV or Excel files
- Predict sentiment for 100+ reviews at once
- Process in seconds

### ✅ Comprehensive Analysis
- Sentiment distribution charts
- Confidence score visualization
- Keyword frequency analysis
- Error pattern identification

### ✅ Model Comparisons
- 6 different architectures tested
- Kernel size impact analysis
- Attention mechanism evaluation
- Performance metrics for each

### ✅ Export Results
- Download as CSV or Excel
- Includes confidence scores
- Error labels if labels provided
- Ready for further analysis

### ✅ Error Analysis
- Confusion matrix visualization
- Accuracy, Precision, Recall, F1
- Identifies false positives/negatives
- Keyword analysis by sentiment

---

## 🔧 Configuration

### Batch App Settings (Sidebar)
```
📁 File Upload
  - Select review column
  - Select label column (optional)

🧹 Data Cleaning
  - Remove null reviews
  - Remove empty reviews  
  - Minimum review length
```

### Model Comparison Options
```bash
--epochs 3              # Training epochs (default: 3)
--batch_size 64         # Batch size (default: 64)
--num_filters 128       # CNN filters (default: 128)
--embedding_dim 100     # Embedding size (default: 100)
--max_len 300           # Max review length (default: 300)
--output_dir outputs    # Output directory
```

---

## 📊 Output Interpretation

### Sentiment Distribution
- Shows % of positive vs negative reviews
- Compares predicted vs actual (if labels provided)

### Confidence Scores
- X-axis: Confidence level (0-100%)
- Y-axis: Number of reviews
- Green: Positive predictions
- Red: Negative predictions

### Top Keywords
- **Left chart**: Most common words in positive reviews
- **Right chart**: Most common words in negative reviews
- **Stop words removed**: Common words like "the", "a", "and"

### Performance Metrics
If labels are provided:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictions that are correct
- **Recall**: Actual positives found
- **F1-Score**: Harmonic mean of precision & recall

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | Run training script first |
| No columns detected | File must have review/text/content/sentence column |
| Slow predictions | Use fewer reviews or reduce batch size |
| Memory error | Reduce batch_size from 64 to 32 in code |
| App won't start | Check if port 8502 is available |

---

## 💡 Example Workflow

### Scenario: Analyze Customer Reviews

1. **Prepare data** (Excel with "Review" column)
   ```
   Review
   "Great product, highly recommend!"
   "Terrible quality, broke after one day"
   "Amazing service and support"
   ```

2. **Upload to batch app**
   ```
   python -m streamlit run batch_app.py
   → Upload file → Select "Review" column → Run
   ```

3. **View results**
   - See 66% positive, 33% negative
   - Download predictions as CSV
   - Share keyword analysis with team

4. **Compare models** (optional)
   ```
   python src/advanced_model_comparison.py
   → View which architecture performs best
   ```

---

## 📈 Performance Benchmarks

On typical hardware (Intel i7, 8GB RAM):

| Task | Time | Notes |
|------|------|-------|
| 10 reviews | <1s | Single batch |
| 100 reviews | 2-3s | Multiple batches |
| 1000 reviews | 20-30s | Full processing |
| Model training | 5-10 min | 5 epochs on SST-2 |
| Model comparison | 15-20 min | 6 architectures × 3 epochs |

---

## 🚀 Next Steps

After running batch predictions:

1. **Export results** - Download CSV/Excel
2. **Share insights** - Use visualization outputs
3. **Compare models** - Run advanced_model_comparison.py
4. **Fine-tune** - Modify hyperparameters
5. **Deploy** - Host batch_app.py on server

---

## 📚 Learn More

- **Full Documentation**: `BATCH_APP_README.md`
- **Training Details**: `src/train_textcnn_imdb.py`
- **Architecture Comparison**: `src/advanced_model_comparison.py`
- **Single Review UI**: `app.py`

---

## ✨ Features Implemented

✅ Batch CSV/Excel file upload
✅ Automatic column detection
✅ Null/empty value handling
✅ Sentiment predictions with confidence scores
✅ Sentiment distribution charts
✅ Keyword frequency analysis
✅ Confusion matrix visualization
✅ Performance metrics (Acc, Prec, Rec, F1)
✅ CSV/Excel export
✅ CNN architecture comparisons
✅ Kernel size impact analysis
✅ Attention mechanism evaluation
✅ Error analysis and reporting

---

**Ready to get started? Run:**
```bash
python -m streamlit run batch_app.py
```

Happy analyzing! 🎬📊
