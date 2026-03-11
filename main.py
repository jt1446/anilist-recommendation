print("this is the main.py!")

from modules.gnn_model import GNNEncoder
from modules.seq_model import RNNPredictor
from modules.trainer import train

# 1. Load Data
data = load_processed_data() 
# 2. Initialize Models
gnn = GNNEncoder()
rnn = RNNPredictor()
# 3. Train
train(gnn, rnn, data)
