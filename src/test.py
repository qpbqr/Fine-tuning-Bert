import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

import config
from model import BERTClass
from dataset import FinancialNewsDataset


def eval_model(model, data_loader, loss_fn, device):
    """
    Evaluate the model
    """
    model = model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss = loss_fn(outputs, targets)
            
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            total_predictions += targets.size(0)
            
            losses.append(loss.item())
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = correct_predictions.double().item() / total_predictions
    avg_loss = np.mean(losses)
    
    return avg_loss, accuracy, all_predictions, all_targets


def evaluate_test_set(model_path=None, test_data_path=None):
    """
    Evaluate model on test set
    
    Args:
        model_path: model path, if None then use config path
        test_data_path: test set path, if None then use config path
    """
    if model_path is None:
        model_path = config.MODEL_PATH
    if test_data_path is None:
        test_data_path = config.TEST_DATA_PATH
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test set file not found: {test_data_path}")
    
    print(f"Loading model: {model_path}")
    print(f"Loading test set: {test_data_path}")
    
    # Load test set
    test_df = pd.read_csv(test_data_path)
    print(f"Test set size: {len(test_df)} samples")
    print(f"Positive samples: {len(test_df[test_df['label']==1])}, Negative samples: {len(test_df[test_df['label']==0])}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create test set dataset and data loader
    test_dataset = FinancialNewsDataset(test_df, tokenizer, config.MAX_LEN)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # Load model
    print(f"\nLoading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    
    # Get number of classes
    num_classes = checkpoint.get('num_classes', 2)
    print(f"Number of classes: {num_classes}")
    
    # Initialize model
    model = BERTClass(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()
    
    print(f"Using device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get model information
    if 'val_acc' in checkpoint:
        print(f"Model validation accuracy: {checkpoint['val_acc']:.4f}")
    if 'epoch' in checkpoint:
        print(f"Training epochs: {checkpoint['epoch']}")
    
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Evaluate model
    print(f"\n{'='*50}")
    print("Evaluating model on test set...")
    print(f"{'='*50}")
    
    test_loss, test_acc, test_preds, test_targets = eval_model(
        model, test_data_loader, loss_fn, config.DEVICE
    )
    
    # Print results
    print(f"\nTest set results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    
    print(f"\nTest set classification report:")
    print(classification_report(test_targets, test_preds, 
                              target_names=['Negative', 'Positive']))
    
    print(f"\nTest set confusion matrix:")
    cm = confusion_matrix(test_targets, test_preds)
    print(cm)
    
    # Calculate additional metrics
    print(f"\nDetailed metrics:")
    print(f"True Positive (TP): {cm[1][1]}")
    print(f"True Negative (TN): {cm[0][0]}")
    print(f"False Positive (FP): {cm[0][1]}")
    print(f"False Negative (FN): {cm[1][0]}")
    
    if cm[1][1] + cm[0][1] > 0:
        precision = cm[1][1] / (cm[1][1] + cm[0][1])
        print(f"Precision: {precision:.4f}")
    
    if cm[1][1] + cm[1][0] > 0:
        recall = cm[1][1] / (cm[1][1] + cm[1][0])
        print(f"Recall: {recall:.4f}")
    
    if cm[1][1] + cm[0][1] > 0 and cm[1][1] + cm[1][0] > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"F1 Score: {f1:.4f}")
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'predictions': test_preds,
        'targets': test_targets,
        'confusion_matrix': cm
    }


if __name__ == "__main__":
    results = evaluate_test_set()