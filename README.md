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
- [ ] write util functions
- [ ] data engineering
    - [x] get data
    - [ ] convert into vectors
    - [ ] make tensors
    - [ ] produce fully cleaned data that can be imported

- [ ] loss/integration
    - [ ] BPR Loss
    - [ ] evaluation script
    - [ ] training loop script

- [ ] gnn
    - [ ] build the graph
    - [ ] create gnn layer
    - [ ] write function to return latent vector for any anime
- [ ] seq model
    - build rnn/attention layer with the input being a sequence of 128 dimension embeds from gnn
    - output is probability distribution of all anime (next recommendation)

- [ ] visualization
    - [ ] build streamlit interface
    - [ ] implement graph viz with PyVis
    - [ ] take the output of the GNN embeddings and create an interactive "Galaxy" of anime where users can click to see clusters
    - [ ] Visualize the "Loss Convergence" curve by reading the training logs

### Features: 
- Uses GNN
- Uses Seq Model
- Makes cool graph visualization (eventually)

### Write-up:
[[google docs link goes here eventually]]
