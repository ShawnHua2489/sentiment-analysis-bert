import os
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
from transformers import AutoTokenizer, AutoModel
from utils.dataset import IMDBDataset
from utils.config import Config

def train():
    # Set device
    paddle.set_device('gpu')
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('models/bert-base-uncased')
    model = AutoModel.from_pretrained('models/bert-base-uncased')
    
    # Create dataset and dataloader
    train_dataset = IMDBDataset(
        data_dir=Config.DATA_DIR,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )
    
    # Define model
    class BertClassifier(nn.Layer):
        def __init__(self, bert_model):
            super(BertClassifier, self).__init__()
            self.bert = bert_model
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(768, 2)
            # Initialize weights with smaller values
            self.classifier.weight.set_value(
                paddle.normal(mean=0.0, std=0.02, shape=self.classifier.weight.shape)
            )
            
        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits
    
    classifier = BertClassifier(model)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        learning_rate=Config.LEARNING_RATE,
        parameters=classifier.parameters(),
        weight_decay=0.01  # L2 regularization
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        classifier.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            # Forward pass
            logits = classifier(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{Config.NUM_EPOCHS}, Loss: {avg_loss:.4f}')
        
        # Early stopping
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            patience_counter = 0
            # Save best model
            paddle.save(classifier.state_dict(), os.path.join(Config.MODEL_DIR, 'best_model.pdparams'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping triggered')
                break
    
    print('Training completed')

if __name__ == '__main__':
    train() 