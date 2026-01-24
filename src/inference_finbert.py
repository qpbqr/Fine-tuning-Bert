import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# FinBERT model for sentiment analysis
# yiyanghkust/finbert-tone is the most popular FinBERT model for sentiment (2019)
FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"


def load_finbert_model(device=None):
    """
    Load FinBERT model for sentiment analysis
    
    Args:
        device: device to load model on, if None auto-select
        
    Returns:
        model: FinBERT model
        tokenizer: FinBERT tokenizer
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading FinBERT model: {FINBERT_MODEL_NAME}")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
    model = model.to(device)
    model.eval()
    
    print("FinBERT model loaded successfully!")
    
    return model, tokenizer


def predict_batch_finbert(model, tokenizer, texts, device=None, batch_size=32, max_length=160):
    """
    Batch prediction using FinBERT
    
    FinBERT-tone has 3 classes: positive (0), negative (1), neutral (2)
    We convert to binary: positive vs negative (ignoring neutral)
    
    Args:
        model: FinBERT model
        tokenizer: FinBERT tokenizer
        texts: list of texts
        device: device
        batch_size: batch size
        max_length: maximum sequence length
        
    Returns:
        predictions: predicted class indices
        probabilities: probability matrix (3 classes: positive, negative, neutral)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # FinBERT outputs logits, apply softmax to get probabilities
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)


def get_positive_probability_finbert(probabilities):
    """
    Convert FinBERT 3-class probabilities to binary positive probability
    
    FinBERT classes: 0=positive, 1=negative, 2=neutral
    We use: P_positive = P(class=0) / (P(class=0) + P(class=1))
    This normalizes to ignore neutral class
    
    Args:
        probabilities: probability matrix with shape (N, 3)
        
    Returns:
        positive_probs: positive probabilities with shape (N,)
    """
    # Extract positive (0) and negative (1) probabilities
    pos_probs = probabilities[:, 0]  # positive
    neg_probs = probabilities[:, 1]  # negative
    
    # Normalize to get P(positive | positive or negative)
    # This ignores neutral class
    total = pos_probs + neg_probs
    # Avoid division by zero
    positive_probs = np.where(total > 0, pos_probs / total, 0.5)
    
    return positive_probs

