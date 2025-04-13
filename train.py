import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from models.bert_classifier import BertClassifier
from utils.data_processing import get_tokenizer, create_data_loader
from config import Config

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

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and tokenizer
    model = BertClassifier().to(device)
    tokenizer = get_tokenizer()
    
    # More diverse dummy data for sentiment analysis
    train_texts = [
        # Positive reviews
        "This movie was absolutely fantastic! I loved every minute of it.",
        "The acting was superb and the plot was engaging from start to finish.",
        "I had a great time watching this film with my friends.",
        "The cinematography was stunning and the story was engaging.",
        "One of the best movies I've seen this year!",
        "The soundtrack was amazing and really added to the experience.",
        "The characters were well-developed and relatable.",
        "I would definitely watch this movie again!",
        
        # Negative reviews
        "The acting was terrible and the plot made no sense.",
        "Waste of time and money, would not recommend.",
        "Boring and predictable, fell asleep halfway through.",
        "The characters were poorly developed and the dialogue was cringe-worthy.",
        "The special effects looked cheap and unrealistic.",
        "The pacing was too slow and the story dragged on.",
        "I couldn't connect with any of the characters.",
        "The ending was disappointing and didn't make sense."
    ]
    train_labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 1 for positive, 0 for negative
    
    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_loader = create_data_loader(
        train_texts,
        train_labels,
        tokenizer,
        Config.BATCH_SIZE,
        Config.MAX_LENGTH
    )
    
    val_loader = create_data_loader(
        val_texts,
        val_labels,
        tokenizer,
        Config.BATCH_SIZE,
        Config.MAX_LENGTH
    )
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)  # Slightly higher learning rate
    total_steps = len(train_loader) * 10  # Increased epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_val_accuracy = 0
    for epoch in range(10):  # Increased epochs
        print(f"\nEpoch {epoch + 1}/10")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"{Config.MODEL_DIR}/bert_classifier.pt")
            print("Saved new best model!")
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

if __name__ == "__main__":
    main() 