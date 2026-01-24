import torch
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train.csv")
VAL_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "val.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "test.csv")
BALANCED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "balanced_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved_weights")
MODEL_PATH = os.path.join(MODEL_DIR, "best_bert_model.bin")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 160

TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5
NUM_WORKERS = 0

RANDOM_SEED = 42