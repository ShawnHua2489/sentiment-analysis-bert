from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
from config import Config

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def get_tokenizer():
    # Use local files from our downloaded model
    return AutoTokenizer.from_pretrained(
        "models/bert-base-uncased",
        local_files_only=True
    )

def create_data_loader(texts, labels, tokenizer, batch_size, max_length):
    dataset = TextClassificationDataset(
        texts=texts,
        labels=labels,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    ) 