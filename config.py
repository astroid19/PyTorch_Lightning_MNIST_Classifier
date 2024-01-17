# Training Hyperparameters
INPUT_SIZE = 784
NUM_CLASSES = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 100

# Dataset 
DATA_DIR = 'dataset/'
NUM_WORKERS = 11


# Compute related 
ACCELERATOR = 'cpu'  # 'gpu
DEVICES = 1  # [0, 1]
PRECISION = 'bf16-mixed'
