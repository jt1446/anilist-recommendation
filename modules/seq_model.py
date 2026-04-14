import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        """
        A stronger GRU-based sequence model with additive attention.

        Args:
            input_dim: Dimension of anime embeddings (e.g., 128)
            hidden_dim: Hidden state size
            output_dim: Number of unique anime (for prediction logits)
            num_layers: Number of GRU layers
            dropout: Dropout probability for regularization
        """
        super(AttentionRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout
        self.num_layers = num_layers

        self.input_projection = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input sequence of embeddings [batch_size, seq_len, input_dim]
            return_attention: whether to return attention weights
        Returns:
            logits: Prediction scores [batch_size, output_dim]
            or (logits, attention_weights) if return_attention=True
        """
        x = self.input_projection(x)
        output, h_n = self.gru(x)

        # Use the last hidden state from the top GRU layer for attention context.
        last_hidden = h_n[-1]
        last_hidden_expanded = last_hidden.unsqueeze(1).expand(-1, output.size(1), -1)

        attention_input = torch.cat([output, last_hidden_expanded], dim=-1)
        weights = self.attention(attention_input).squeeze(-1)
        weights = F.softmax(weights, dim=1)

        context = torch.sum(weights.unsqueeze(-1) * output, dim=1)
        context = self.layer_norm(context + last_hidden)
        context = self.dropout(context)

        logits = self.fc(context)
        if return_attention:
            return logits, weights
        return logits


def predict_next_anime(model, user_history_embeddings):
    """
    Takes a sequence of vectors and outputs the predicted next vector.
    Args:
        model: Trained AttentionRNN model
        user_history_embeddings: Tensor [1, seq_len, input_dim]
    Returns:
        prediction_logits: Probability scores for every anime in the database [1, output_dim]
    """
    model.eval()
    with torch.no_grad():
        logits = model(user_history_embeddings)
    return logits
