class Config:
    # Model parameters
    MODEL_NAME = "models/bert-base-uncased"  # Using local model path
    MAX_LENGTH = 128
    BATCH_SIZE = 32  # Increased batch size
    NUM_LABELS = 2  # Binary classification by default
    
    # Training parameters
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 100
    
    # Data parameters
    TRAIN_SIZE = 0.8
    VAL_SIZE = 0.1
    TEST_SIZE = 0.1
    SUBSET_SIZE = 2000  # Use a smaller subset for faster training
    
    # Paths
    DATA_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    MODEL_DIR = "models" 