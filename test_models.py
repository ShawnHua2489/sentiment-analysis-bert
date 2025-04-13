from transformers import AutoTokenizer, AutoModel
import torch
import os

def test_model(model_path):
    print("\nTesting BERT model...")
    print(f"Model path: {model_path}")
    print(f"Path exists: {os.path.exists(model_path)}")
    
    try:
        # Test tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        print("Tokenizer loaded successfully!")
        
        # Test model
        print("Loading model...")
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        print("Model loaded successfully!")
        
        # Test inference
        print("Testing inference...")
        text = "This is a test sentence to verify the model works."
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        print("Inference successful!")
        print(f"Output shape: {outputs.last_hidden_state.shape}")
        return True
        
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Current working directory:", os.getcwd())
    
    # Test BERT
    bert_path = "./models/bert-base-uncased"
    print("\nChecking BERT files:")
    if os.path.exists(bert_path):
        print("BERT directory contents:", os.listdir(bert_path))
    works = test_model(bert_path)
    
    print("\nTest Summary:")
    print(f"BERT: {'✓' if works else '✗'}") 