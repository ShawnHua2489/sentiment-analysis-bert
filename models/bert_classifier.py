import torch
import torch.nn as nn
from transformers import AutoModel
from config import Config

class BertClassifier(nn.Module):
    def __init__(self, config=Config):
        super(BertClassifier, self).__init__()
        self.config = config
        
        # Load pre-trained BERT model from local files
        self.bert = AutoModel.from_pretrained(
            "models/bert-base-uncased",
            local_files_only=True
        )
        
        # Add dropout with higher rate for regularization
        self.dropout = nn.Dropout(0.3)  # Increased from 0.1 to 0.3
        
        # Classification head with L2 regularization
        self.classifier = nn.Linear(768, config.NUM_LABELS)
        
        # Initialize weights with smaller values
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
        
    def forward(self, input_ids, attention_mask):
        # Get the last hidden state from BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token's representation for classification
        pooled_output = outputs[0][:, 0]
        
        # Apply dropout (regularization)
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        return logits 