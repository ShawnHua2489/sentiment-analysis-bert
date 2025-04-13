# BERT Text Classification

A text classification project using BERT (Bidirectional Encoder Representations from Transformers) for sentiment analysis. This project demonstrates how to:
- Download and use pre-trained BERT models
- Fine-tune BERT for text classification
- Implement a custom classification head
- Train and evaluate the model

## Project Structure

```
text_classification/
├── config.py           # Configuration settings
├── models/
│   └── bert_classifier.py  # Custom BERT classifier implementation
├── utils/
│   └── data_processing.py  # Data processing utilities
├── train.py           # Training script
├── test_models.py     # Model testing script
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd text_classification
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the BERT model:
```bash
git clone https://huggingface.co/bert-base-uncased models/bert-base-uncased
```

## Usage

### Training

To train the model with the provided dummy data:
```bash
python train.py
```

The script will:
- Load the BERT model
- Train on the dummy sentiment analysis dataset
- Save the best model based on validation accuracy
- Print training metrics and classification reports

### Testing

To test the model:
```bash
python test_models.py
```

## Model Architecture

The project uses a custom BERT classifier that:
1. Uses pre-trained BERT base uncased model
2. Adds a classification head with dropout
3. Uses the [CLS] token representation for classification

## Configuration

Key parameters can be modified in `config.py`:
- `MODEL_NAME`: Path to the BERT model
- `MAX_LENGTH`: Maximum sequence length
- `BATCH_SIZE`: Training batch size
- `NUM_LABELS`: Number of classification labels
- `LEARNING_RATE`: Learning rate for training
- `NUM_EPOCHS`: Number of training epochs

## Dependencies

- torch>=1.9.0
- transformers>=4.11.0
- datasets>=1.11.0
- numpy>=1.19.5
- pandas>=1.3.0
- scikit-learn>=0.24.2
- tqdm>=4.62.3

## License

This project is licensed under the MIT License - see the LICENSE file for details. 