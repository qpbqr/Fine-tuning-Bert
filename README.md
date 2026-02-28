# Financial News Sentiment Analysis Fine-tuning Project

## Motivation

Previously, I attempted to use pre-trained models for factor construction, but discovered that strategies often performed better before the model's release date. Although we only use these models for semantic understanding, this still introduced look-ahead bias during inference. To avoid this issue, I decided to perform fine-tuning myself. The results are modest, but the process is quite interesting.

---

This project fine-tunes BERT models for sentiment analysis on financial news headlines, designed to identify implicit sentiment factors in analyst reports. The project uses PyTorch and Transformers libraries to implement BERT fine-tuning, supporting binary sentiment classification (positive/negative).

## Project Overview

This project aims to analyze sentiment tendencies in financial news headlines by fine-tuning BERT models, helping to identify implicit sentiment information in analyst reports. The project includes complete data processing, model training, evaluation, and inference pipelines.

## Project Structure

```
Sentiment Finetune/
├── data/
│   └── processed/          # Processed data
│       ├── balanced_data.csv      # Balanced dataset
│       ├── train.csv              # Training set
│       ├── val.csv                # Validation set
│       ├── test.csv               # Test set
│       └── *.csv                  # Statistics files
├── models/
│   └── saved_weights/      # Saved model weights
│       └── best_bert_model.bin
├── logs/                   # Log files
├── notebooks/              # Jupyter notebooks
│   ├── eda.ipynb          # Exploratory data analysis
│   ├── preprocessing.ipynb # Data preprocessing
│   └── temp_processing/   # Temporary processing files
└── src/                    # Source code
    ├── config.py          # Configuration file
    ├── model.py           # Model definition
    ├── dataset.py         # Dataset class
    ├── train.py           # Training script
    ├── test.py            # Testing script
    ├── inference.py       # Inference script (fine-tuned BERT)
    └── inference_finbert.py # FinBERT inference script
```

## Requirements

- Python
- PyTorch
- transformers
- pandas
- numpy
- scikit-learn
- tqdm

## Installation

1. Clone or download the project to your local machine

2. Install dependencies:

```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

## Data Format

The data files used in this project should contain the following columns:
- `headline`: Financial news headline (text)
- `label`: Sentiment label (0=negative, 1=positive)
- `ticker`: Stock ticker symbol (used for data splitting)

Data files should be saved in the `data/processed/` directory.

## Usage

### 1. Data Splitting

First, split the balanced data by ticker into training, validation, and test sets:

```bash
python src/split_data.py
```

This script will:
- Group data by ticker
- For each ticker, use 60% of news for training, and split the remaining 40% equally between validation and test sets
- Maintain balanced positive and negative samples in the training set
- Generate `train.csv`, `val.csv`, `test.csv`, and statistics files

### 2. Model Training

Train the BERT model:

```bash
python src/train.py
```

Training configuration can be modified in `src/config.py`:
- `EPOCHS`: Number of training epochs (default: 5)
- `TRAIN_BATCH_SIZE`: Training batch size (default: 32)
- `LEARNING_RATE`: Learning rate (default: 1e-5)
- `MAX_LEN`: Maximum sequence length (default: 160)

During training, the script will:
- Automatically save the best model (based on validation accuracy)
- Record training history to `models/saved_weights/training_history.json`
- Display training and validation metrics for each epoch

### 3. Model Testing

Evaluate model performance on the test set:

```bash
python src/test.py
```

This script will output:
- Test set accuracy and loss
- Classification report (precision, recall, F1 score)
- Confusion matrix
- Detailed metrics (TP, TN, FP, FN)

### 4. Model Inference

#### Using Fine-tuned BERT Model

```python
from src.inference import load_model, predict_single, predict_batch, predict_from_csv

# Load model
model, tokenizer = load_model()

# Single text prediction
text = "Company reports strong quarterly earnings"
pred, probs = predict_single(model, tokenizer, text)
print(f"Predicted class: {'positive' if pred == 1 else 'negative'}")
print(f"Probabilities: negative={probs[0]:.4f}, positive={probs[1]:.4f}")

# Batch prediction
texts = ["Text 1", "Text 2", ...]
predictions, probabilities = predict_batch(model, tokenizer, texts)

# Predict from CSV file
df = predict_from_csv(
    model, tokenizer, 
    csv_path="path/to/data.csv",
    text_column="headline",
    output_path="path/to/output.csv"
)
```

#### Using FinBERT Model

The project also provides an inference script using the pre-trained FinBERT model:

```python
from src.inference_finbert import load_finbert_model, predict_batch_finbert, get_positive_probability_finbert

# Load FinBERT model
model, tokenizer = load_finbert_model()

# Batch prediction (returns 3 classes: positive, negative, neutral)
texts = ["Text 1", "Text 2", ...]
predictions, probabilities = predict_batch_finbert(model, tokenizer, texts)

# Convert to binary classification probability (ignoring neutral class)
positive_probs = get_positive_probability_finbert(probabilities)
```

## Model Architecture

The project uses a BERT-based classification model:

- **Base Model**: `bert-base-uncased`
- **Architecture**: BERT + Dropout(0.3) + Linear classification layer
- **Output Dimension**: 768 → num_classes (default: 2 classes)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW
- **Gradient Clipping**: max_norm=1.0

## Configuration

Main configuration is in `src/config.py`:

```python
MODEL_NAME = "bert-base-uncased"  # BERT model name
MAX_LEN = 160                      # Maximum sequence length
TRAIN_BATCH_SIZE = 32             # Training batch size
VALID_BATCH_SIZE = 16             # Validation batch size
EPOCHS = 5                        # Number of training epochs
LEARNING_RATE = 1e-5             # Learning rate
RANDOM_SEED = 42                  # Random seed
```

## Data Splitting Strategy

Data splitting uses a ticker-based grouping approach to ensure:
- Training set: 60% of negative news per ticker + equal amount of positive news
- Validation set: 50% of remaining negative news per ticker
- Test set: 50% of remaining negative news per ticker
- Positive news in validation and test sets are allocated proportionally

This strategy prevents data leakage, ensuring that data from the same ticker does not appear in both training and test sets simultaneously.

## Output Files

- **Model File**: `models/saved_weights/best_bert_model.bin`
- **Training History**: `models/saved_weights/training_history.json`
- **Data Statistics**: `data/processed/split_stats.csv`

## Notes

1. The BERT pre-trained model (approximately 440MB) will be automatically downloaded on first run
2. GPU is recommended for faster training, CPU training is slower
3. Data file paths need to be correctly configured in `config.py`
4. Ensure sufficient disk space for storing models and data

## Features

- Support for custom model paths and parameters
- Support for batch inference and CSV file batch processing
- FinBERT provided as an alternative model
- Complete training history recording and evaluation metrics

## License

This project is for learning and research purposes only.

## Contact

For questions or suggestions, please submit an Issue through the project repository.
