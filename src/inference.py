import torch
import torch.nn as nn
from transformers import BertTokenizer
import pandas as pd
import numpy as np
import os

import config
from model import BERTClass


def load_model(model_path=None, num_classes=2, device=None):
    """
    Load trained model
    
    Args:
        model_path: model path, if None then use config path
        num_classes: number of classes
        device: device, if None then use config device
        
    Returns:
        model: loaded model
        tokenizer: tokenizer
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    
    if device is None:
        device = config.DEVICE
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # if there is saved number of classes, use it
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    
    # initialize model
    model = BERTClass(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    
    val_acc = checkpoint.get('val_acc', 'N/A')
    if isinstance(val_acc, (int, float)):
        print(f"Model loaded successfully! Validation accuracy: {val_acc:.4f}")
    else:
        print(f"Model loaded successfully!")
    
    return model, tokenizer


def predict_single(model, tokenizer, text, device=None):
    """
    Predict single text
    
    Args:
        model: trained model
        tokenizer: tokenizer
        text: input text
        device: device
        
    Returns:
        prediction: predicted class (0=negative, 1=positive)
        probabilities: probabilities of each class
    """
    if device is None:
        device = config.DEVICE
    
    # preprocess text
    text = str(text)
    text = " ".join(text.split())
    
    # Tokenize
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=config.MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    # move to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
    
    return prediction, probabilities.cpu().numpy()[0]


def predict_batch(model, tokenizer, texts, device=None, batch_size=32):
    """
    Batch prediction
    
    Args:
        model: trained model
        tokenizer: tokenizer
        texts: text list
        device: device
        batch_size: batch size
        
    Returns:
        predictions: predicted results list
        probabilities: probability matrix
    """
    if device is None:
        device = config.DEVICE
    
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        encoded = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)


def predict_from_csv(model, tokenizer, csv_path, text_column='headline', 
                     output_path=None, device=None):
    """
    Read data from CSV file and predict
    
    Args:
        model: trained model
        tokenizer: tokenizer
        csv_path: CSV file path
        text_column: text column name
        output_path: output file path, if None then not save
        device: device
        
    Returns:
        df: DataFrame containing predicted results
    """
    df = pd.read_csv(csv_path)
    
    print(f"Reading {len(df)} data...")
    
    texts = df[text_column].tolist()
    predictions, probabilities = predict_batch(model, tokenizer, texts, device)
    
    # add predicted results to DataFrame
    df['prediction'] = predictions
    df['probability'] = probabilities.max(axis=1)
    df['sentiment'] = df['prediction'].map({0: 'negative', 1: 'positive'})
    
    # add probabilities of each class
    for i in range(probabilities.shape[1]):
        df[f'prob_class_{i}'] = probabilities[:, i]
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Predicted results saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # example: load model and predict
    model, tokenizer = load_model()
    
    # single prediction example
    test_texts = [
        "This is a positive news about the company.",
        "The company faces significant challenges ahead."
    ]
    
    for text in test_texts:
        pred, probs = predict_single(model, tokenizer, text)
        sentiment = "positive" if pred == 1 else "negative"
        print(f"\nText: {text}")
        print(f"Predicted class: {sentiment} (class ID: {pred})")
        print(f"Negative probability: {probs[0]:.4f}, Positive probability: {probs[1]:.4f}")