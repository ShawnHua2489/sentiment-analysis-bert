import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import random

# Add persistent path to Python path
sys.path.append('/home/aistudio/external-libraries')

from models.bert_classifier import BertClassifier
from utils.data_processing import get_tokenizer, create_data_loader
from config import Config

# Baidu AI Studio paths
datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/data108104/'  # Updated with your dataset ID
output_dir = "/root/paddlejob/workspace/output/"

def load_baidu_dataset(dataset_path):
    """Load dataset from Baidu AI Studio format"""
    print("Loading dataset from Baidu AI Studio...")
    print(f"Dataset path: {dataset_path}")
    
    # List files in the dataset directory
    print("Files in dataset directory:")
    print(os.listdir(dataset_path))
    
    # Assuming the dataset is in CSV format
    train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
    
    # Extract texts and labels
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    return train_texts, train_labels, test_texts, test_labels

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        
        # Gradient clipping for regularization
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=['Negative', 'Positive'], zero_division=0))
    return total_loss / len(data_loader), accuracy

def plot_training_curves(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def get_subset(texts, labels, size):
    indices = list(range(len(texts)))
    random.shuffle(indices)
    subset_indices = indices[:size]
    return [texts[i] for i in subset_indices], [labels[i] for i in subset_indices]

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and tokenizer
    model = BertClassifier().to(device)
    tokenizer = get_tokenizer()
    
    # Load dataset from Baidu AI Studio
    # Get the dataset path from command line argument or use default
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else datasets_prefix
    train_texts, train_labels, test_texts, test_labels = load_baidu_dataset(dataset_path)
    
    # Get subset for faster training
    train_texts, train_labels = get_subset(
        train_texts,
        train_labels,
        Config.SUBSET_SIZE
    )
    test_texts, test_labels = get_subset(
        test_texts,
        test_labels,
        Config.SUBSET_SIZE // 2
    )
    
    print(f"Using {len(train_texts)} training examples and {len(test_texts)} test examples")
    
    # Create data loaders
    train_loader = create_data_loader(
        train_texts,
        train_labels,
        tokenizer,
        Config.BATCH_SIZE,
        Config.MAX_LENGTH
    )
    
    test_loader = create_data_loader(
        test_texts,
        test_labels,
        tokenizer,
        Config.BATCH_SIZE,
        Config.MAX_LENGTH
    )
    
    # Initialize optimizer with weight decay (L2 regularization)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * Config.NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop with early stopping
    best_val_accuracy = 0
    patience = 3
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        train_losses.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_accuracy = evaluate(model, test_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, 'bert_classifier.pt'))
            print("Saved new best model!")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, val_accuracies)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    main() 