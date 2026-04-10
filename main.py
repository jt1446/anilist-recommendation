print("starting backend...")

from modules.gnn_model import AnimeGNN
from modules.trainer import SeqRNNModel, main as train_models
from modules.utils import load_processed_data

# Entry point for running the training pipeline
if __name__ == "__main__":
    print("Starting training pipeline...")
    train_models()
