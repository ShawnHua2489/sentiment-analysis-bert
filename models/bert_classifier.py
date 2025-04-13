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
        
        # Classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, config.NUM_LABELS)  # 768 is the hidden size of BERT
        
    def forward(self, input_ids, attention_mask):
        # Get the last hidden state from BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use the [CLS] token's representation for classification
        pooled_output = outputs[0][:, 0]
        
        # Apply dropout and classification head
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits 