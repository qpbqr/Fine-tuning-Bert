import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import json

import config
from model import BERTClass
from dataset import FinancialNewsDataset

def train_epoch(model, data_loader, loss_fn, optimizer, device, epoch):
    """
    Train one epoch
    """
    model = model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['mask'].to(device)
        targets = batch['targets'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(torch.argmax(outputs, dim=1) == targets)
        total_predictions += targets.size(0)
        
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        # update progress bar
        progress_bar.set_postfix({
            'loss': np.mean(losses),
            'acc': correct_predictions.double().item() / total_predictions
        })
    
    return np.mean(losses), correct_predictions.double().item() / total_predictions


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


def run_training():
    """
    Main training function
    """
    # set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
    
    # load data
    print("loading data...")
    train_df = pd.read_csv(config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(config.VAL_DATA_PATH)
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Validation set: {len(val_df)} rows")
    
    # initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    
    # create dataset
    train_dataset = FinancialNewsDataset(train_df, tokenizer, config.MAX_LEN)
    val_dataset = FinancialNewsDataset(val_df, tokenizer, config.MAX_LEN)
    
    # create data loader
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    
    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )
    
    # determine number of classes
    num_classes = len(train_df['label'].unique())
    print(f"Number of classes: {num_classes}")
    
    # initialize model
    print("Initializing model...")
    model = BERTClass(num_classes=num_classes)
    model = model.to(config.DEVICE)
    
    print(f"Using device: {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # training loop
    best_accuracy = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\nStarting training, device: {config.DEVICE}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Training configuration: Epochs={config.EPOCHS}, Batch Size={config.TRAIN_BATCH_SIZE}, LR={config.LEARNING_RATE}")
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.EPOCHS}")
        print(f"{'='*50}")
        
        # train
        train_loss, train_acc = train_epoch(
            model, train_data_loader, loss_fn, optimizer, config.DEVICE, epoch
        )
        
        # validate
        val_loss, val_acc, val_preds, val_targets = eval_model(
            model, val_data_loader, loss_fn, config.DEVICE
        )
        
        # record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\ntraining - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"validating - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            print(f"Saving best model (Validation accuracy: {best_accuracy:.4f})")
            
            # ensure directory exists
            os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'config': {
                    'model_name': config.MODEL_NAME,
                    'max_len': config.MAX_LEN,
                    'batch_size': config.TRAIN_BATCH_SIZE,
                    'learning_rate': config.LEARNING_RATE,
                }
            }, config.MODEL_PATH)
        
        # print validation set classification report
        if epoch == config.EPOCHS:
            print("\nValidation set classification report:")
            print(classification_report(val_targets, val_preds, 
                                      target_names=['Negative', 'Positive']))
            print("\nValidation set confusion matrix:")
            print(confusion_matrix(val_targets, val_preds))
    
    # save training history
    history_path = os.path.join(os.path.dirname(config.MODEL_PATH), "training_history.json")
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\nTraining history saved to: {history_path}")
    
    print(f"\nTraining completed! Best validation accuracy: {best_accuracy:.4f}")
    print(f"Model saved to: {config.MODEL_PATH}")
    
    return history


if __name__ == "__main__":
    history = run_training()