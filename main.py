print("starting backend...")

from modules.gnn_model import GNNEncoder
from modules.seq_model import RNNPredictor
from modules.trainer import train
from modules.utils import load_processed_data

# 1. Load Data
data = load_processed_data()
print("Data loaded successfully!")
# 2. Initialize Models
gnn = GNNEncoder()
print("GNN model initialized.")
rnn = RNNPredictor()
print("RNN model initialized.")
# 3. Train
train(gnn, rnn, data)
print("Training completed!")
