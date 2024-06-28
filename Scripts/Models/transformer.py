import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from dotenv import load_dotenv
import math

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

SIGNAL_LENGTH = int(os.getenv('SAMPLES_PER_SEGMENT'))
class Attention(nn.Module):
    def __init__(self, key_size, dropout=None):
        super().__init__()
        self.scaling = (key_size ** 0.5)
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scaling

        if mask is not None:
            # Set the masked elements to a very low value - per Vaswani et al.
            scores = scores.masked_fill(mask == 0, -1e9)

        probs = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            probs = self.dropout(probs)

        return torch.matmul(probs, V), probs

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_len=SIGNAL_LENGTH, dropout=None):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        encoding = torch.zeros((1, max_len, embedding_size)).float()
        encoding.require_grad = False

        # SIN/COS positional encoding per Vaswani et al.
        position = torch.arange(0, max_len).float().reshape(-1,1)
        period = torch.pow(10000.0, torch.arange(0, embedding_size, 2).float() / embedding_size)

        encoding[:, :, 0::2] = torch.sin(position / period)
        encoding[:, :, 1::2] = torch.cos(position / period)
        self.register_buffer('encoding', encoding)

    def forward(self, X):
        X = X + self.encoding[:, :X.size(1), :]
        return self.dropout(X) if self.dropout is not None else X

class PositionwiseFeedForward(nn.Module):
    def __init__(self, embedding_size, feature_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(embedding_size, feature_size)
        self.w_2 = nn.Linear(feature_size, embedding_size)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class MultiHeadedAttention(nn.Module):
    def __init__(self, head_amount, embedding_size, bias=False, dropout=0.1):
        super().__init__()
        assert embedding_size % head_amount == 0

        # We assume key_dim == value_dim == query_dim
        self.embedding_size = embedding_size
        self.key_size = embedding_size // head_amount
        self.head_amount = head_amount

        # K/Q/V linear transformations
        self.query_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.key_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.value_linear = nn.Linear(embedding_size, embedding_size, bias=bias)

        self.output_linear = nn.Linear(embedding_size, embedding_size)
        self.attention = Attention(self.key_size, dropout=dropout)

    def forward(self, query, key, value, mask=None):
        query_embedding = self.split_embedding(self.query_linear(query))
        key_embedding = self.split_embedding(self.key_linear(key))
        value_embedding = self.split_embedding(self.value_linear(value))

        X, _ = self.attention(query_embedding, key_embedding, value_embedding, mask=mask)

        return self.output_linear(self.combine_embedding(X))

    def split_embedding(self, x):
        batch_size, qkv_amount, _ = x.size()
        x = x.view(batch_size, qkv_amount, self.head_amount, self.key_size)
        return x.transpose(1, 2)

    def combine_embedding(self, x):
        batch_size, _, qkv_amount, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, qkv_amount, self.head_amount * self.key_size)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, head_amount, embedding_size, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.self_attention = MultiHeadedAttention(head_amount, embedding_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(embedding_size, feature_size=embedding_size, dropout=dropout)

        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        X = self.layer_norm1(X + self.dropout1(self.self_attention(X, X, X, mask=mask)))
        X = self.layer_norm2(X + self.dropout2(self.feed_forward(X)))
        return X

class TransformerEncoder(nn.Module):
    def __init__(self, head_amount=8, embedding_size=128, layer_amount=6, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.position = PositionalEncoding(embedding_size, dropout=dropout)
        self.layers = nn.ModuleList([TransformerEncoderBlock(head_amount, embedding_size, dropout) for _ in range(layer_amount)])

    def forward(self, X, mask=None):
        X = self.position(X)
        for layer in self.layers:
            X = layer(X, mask=mask)
        return X

if __name__ == "__main__":
    bert = TransformerEncoder()
    input = torch.FloatTensor(torch.randn(19,100,128))
    X = bert(input)
    print(X.shape)