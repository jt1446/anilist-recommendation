# Anilist Recommendation Algorithm
## 📦 Data Access

### Raw Data (JSON files)
The complete raw dataset (75 users, 61,693 entries, 217MB) is available on Google Drive:
[**Download raw_data.zip**](https://drive.google.com/file/d/1x6jxUVodP0qFMnS6pqWlXBjhJhVraO9H/view?usp=drive_link)

After downloading, place the zip file in your project folder and extract:
```bash
unzip raw_data_backup.zip
```

### Goal:
Making a recommendation algorithm that takes a user's lists and outputs recommendations, predictions of what they will watch next, etc.

### Todos:
- [x] write util functions
- [ ] data engineering
    - [x] get data
    - [x] convert into vectors
    - [x] make tensors
    - [x] produce fully cleaned data that can be imported

- [x] loss/integration
    - [x] BPR Loss
    - [x] evaluation script (Metrics: Hit@10: 0.07, MRR: 0.02)
    - [x] training loop script

- [x] gnn
    - [x] build the graph
    - [x] create gnn layer
    - [x] write function to return latent vector for any anime
- [x] seq model
    - build rnn/attention layer with the input being a sequence of 128 dimension embeds from gnn
    - output is probability distribution of all anime (next recommendation)

- [x] visualization
    - [x] build streamlit interface
    - [x] implement graph viz with Plotly/UMAP (replacing PyVis for better performance)
    - [x] take the output of the GNN embeddings and create an interactive "Galaxy" of anime where users can click to see clusters
    - [x] Visualize the "Loss Convergence" curve by reading the training logs

### Features: 
- Uses GNN
- Uses Seq Model
- Makes cool graph visualization (eventually)

### Write-up:
[[google docs link goes here eventually]]
