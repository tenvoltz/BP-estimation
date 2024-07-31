import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from dotenv import load_dotenv
import math
from einops.layers.torch import Rearrange

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

SIGNAL_LENGTH = int(os.getenv('INPUT_LENGTH'))

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class SequencePooling1D(nn.Module):
    def __init__(self, reduction=1):
        super(SequencePooling1D, self).__init__()
        self.pool = nn.Sequential(
            Rearrange("batch sequence embedding -> batch embedding sequence"),
            nn.MaxPool1d(reduction) if reduction > 1 else nn.Identity(),
            Rearrange("batch embedding sequence -> batch sequence embedding"),
        )
    def forward(self, x):
        return self.pool(x)
class SimpleAttention(nn.Module):
    def __init__(self, sequence_length, dropout=None):
        super().__init__()
        self.scaling = sequence_length
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

    def forward(self, Q, K, V, mask=None):
        # Galerkin attention - per Cao et al.
        scores = torch.matmul(K.transpose(-2, -1), V)
        if mask is not None:
            print("No mask support for simple attention")
        probs = scores / self.scaling
        if self.dropout is not None:
            probs = self.dropout(probs)
        return torch.matmul(Q, probs), probs
class SimpleMultiHeadedAttention(nn.Module):
    def __init__(self, sequence_length, head_amount, embedding_size, bias=False, dropout=0.1):
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

        self.key_norm = nn.LayerNorm(embedding_size)
        self.value_norm = nn.LayerNorm(embedding_size)

        self.output_linear = nn.Linear(embedding_size, embedding_size)
        self.attention = SimpleAttention(sequence_length, dropout=dropout)

    def forward(self, query, key, value, mask=None):
        query_embedding = self.split_embedding(self.query_linear(query))
        key_embedding = self.split_embedding(self.key_norm(self.key_linear(key)))
        value_embedding = self.split_embedding(self.value_norm(self.value_linear(value)))

        X, _ = self.attention(query_embedding, key_embedding, value_embedding, mask=mask)

        return self.output_linear(self.combine_embedding(X))

    def split_embedding(self, x):
        batch_size, qkv_amount, _ = x.size()
        x = x.view(batch_size, qkv_amount, self.head_amount, self.key_size)
        return x.transpose(1, 2)

    def combine_embedding(self, x):
        batch_size, _, qkv_amount, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, qkv_amount, self.head_amount * self.key_size)
class SimpleTransformerEncoderBlock(nn.Module):
    def __init__(self, sequence_length, head_amount, embedding_size, dropout=0.1):
        super(SimpleTransformerEncoderBlock, self).__init__()

        self.self_attention = SimpleMultiHeadedAttention(sequence_length, head_amount, embedding_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(embedding_size, feature_size=embedding_size, dropout=dropout)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        X = X + self.dropout1(self.self_attention(X, X, X, mask=mask))
        X = X + self.dropout2(self.feed_forward(X))
        return X
class SimpleTransformerEncoder(nn.Module):
    def __init__(self, sequence_length=512, head_amount=8, embedding_size=128, layer_amount=6, dropout=0.1):
        super(SimpleTransformerEncoder, self).__init__()
        self.position = PositionalEncoding(embedding_size, dropout=dropout)
        self.layers = nn.ModuleList([SimpleTransformerEncoderBlock(sequence_length, head_amount, embedding_size, dropout)
                                     for _ in range(layer_amount)])

    def forward(self, X, mask=None):
        X = self.position(X)
        for layer in self.layers:
            X = layer(X, mask=mask)
        return X

class LearnableScalingAttention(nn.Module):
    def __init__(self, key_size, dropout=None):
        super().__init__()
        self.scaling = nn.Parameter(torch.sqrt(torch.tensor(key_size)))
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * 1 / self.scaling

        if mask is not None:
            # Set the masked elements to a very low value - per Vaswani et al.
            scores = scores.masked_fill(mask == 0, -1e9)

        probs = F.softmax(scores, dim=-1)

        if self.dropout is not None:
            probs = self.dropout(probs)

        return torch.matmul(probs, V), probs

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
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
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
    def __init__(self, head_amount, embedding_size, bias=False, mask=None, dropout=0.1, attention=Attention):
        super().__init__()
        assert embedding_size % head_amount == 0
        self.register_buffer('mask', mask)

        # We assume key_dim == value_dim == query_dim
        self.embedding_size = embedding_size
        self.key_size = embedding_size // head_amount
        self.head_amount = head_amount

        # K/Q/V linear transformations
        self.query_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.key_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.value_linear = nn.Linear(embedding_size, embedding_size, bias=bias)

        self.output_linear = nn.Linear(embedding_size, embedding_size)
        self.attention = attention(self.key_size, dropout=dropout)

    def forward(self, X):
        query_embedding = self.split_embedding(self.query_linear(X))
        key_embedding = self.split_embedding(self.key_linear(X))
        value_embedding = self.split_embedding(self.value_linear(X))

        X, _ = self.attention(query_embedding, key_embedding, value_embedding, mask=self.mask)

        return self.output_linear(self.combine_embedding(X))

    def split_embedding(self, x):
        batch_size, qkv_amount, _ = x.size()
        x = x.view(batch_size, qkv_amount, self.head_amount, self.key_size)
        return x.transpose(1, 2)

    def combine_embedding(self, x):
        batch_size, _, qkv_amount, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, qkv_amount, self.head_amount * self.key_size)

class ResizeFeedForward(nn.Module):
    def __init__(self, input_size, feature_size, output_size, dropout=0.1):
        super(ResizeFeedForward, self).__init__()
        self.w_1 = nn.Linear(input_size, feature_size)
        self.w_2 = nn.Linear(feature_size, output_size)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
class MultiHeadedPoolingAttention(nn.Module):
    def __init__(self, head_amount, embedding_size, reduction_ratio=2, bias=False, dropout=0.1):
        super().__init__()
        assert (embedding_size // reduction_ratio) % head_amount == 0

        # We assume key_dim == value_dim == query_dim
        self.key_size = embedding_size // head_amount
        self.head_amount = head_amount

        # K/Q/V linear transformations
        self.query_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.key_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.value_linear = nn.Linear(embedding_size, embedding_size, bias=bias)
        
        self.pools = nn.ModuleList([nn.MaxPool1d(kernel_size=reduction_ratio, stride=reduction_ratio) 
                                       for _ in range(4)])

        self.output_linear = nn.Linear(embedding_size, embedding_size)
        self.attention = Attention(self.key_size, dropout=dropout)

    def forward(self, X, mask=None):
        query_residual = self.pooling(self.pools[0], self.query_linear(X))
        query_embedding = self.split_embedding(query_residual)
        key_embedding = self.split_embedding(self.pooling(self.pools[1],self.key_linear(X)))
        value_embedding = self.split_embedding(self.pooling(self.pools[2],self.value_linear(X)))
        residual = self.pooling(self.pools[3],X)
        X, _ = self.attention(query_embedding, key_embedding, value_embedding, mask=mask)

        X = self.combine_embedding(X)
        return self.output_linear(X + query_residual) + residual

    def pooling(self, pool, x):
        x = x.permute(0, 2, 1)
        x = pool(x)
        x = x.permute(0, 2, 1)
        return x

    def split_embedding(self, x):
        batch_size, qkv_amount, _ = x.size()
        x = x.view(batch_size, qkv_amount, self.head_amount, self.key_size)
        return x.transpose(1, 2)

    def combine_embedding(self, x):
        batch_size, _, qkv_amount, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, qkv_amount, self.head_amount * self.key_size)

class MultiBranchMHSA(nn.Module):
    def __init__(self, branch_amount, head_amount, embedding_size, dropout=0.1):
        super().__init__()
        self.branches = nn.ModuleList([MultiHeadedAttention(head_amount, embedding_size, dropout=dropout)
                                       for _ in range(branch_amount)])


    def forward(self, X, mask=None):
        # Get the average attention from all branches
        X = torch.stack([branch(X, mask=mask) for branch in self.branches], dim=0).mean(dim=0)
        return X

class MetaFormerEncoderBlock(nn.Module):
    def __init__(self, channel_mixer, feature_mixer, sequence_length, embedding_size, dropout=0.1):
        super(MetaFormerEncoderBlock, self).__init__()

        self.channel_mixer = channel_mixer
        self.feature_mixer = feature_mixer

        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        #self.dropPath1 = DropPath(0.1)
        #self.dropPath2 = DropPath(0.1)

    def forward(self, X):
        #X = self.layer_norm1(X + self.dropout1(self.channel_mixer(X)))
        #X = self.layer_norm2(X + self.dropout2(self.feature_mixer(X)))
        # Pre-LN
        #X = X + self.dropPath1(self.dropout1(self.channel_mixer(self.layer_norm1(X))))
        #X = X + self.dropPath2(self.dropout2(self.feature_mixer(self.layer_norm2(X))))
        X = X + self.dropout1(self.channel_mixer(self.layer_norm1(X)))
        X = X + self.dropout2(self.feature_mixer(self.layer_norm2(X)))
        return X


class TransformerEncoderBlock(nn.Module):
    def __init__(self, head_amount, embedding_size, dropout=0.1, mlp_dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()

        self.self_attention = MultiHeadedAttention(head_amount, embedding_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(embedding_size, feature_size=embedding_size, dropout=mlp_dropout)

        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.layer_norm2 = nn.LayerNorm(embedding_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        X = self.layer_norm1(X + self.dropout1(self.self_attention(X, mask=mask)))
        X = self.layer_norm2(X + self.dropout2(self.feed_forward(X)))
        # Pre-LN
        #X = X + self.dropout1(self.self_attention(self.layer_norm1(X), mask=mask))
        #X = X + self.dropout2(self.feed_forward(self.layer_norm2(X)))
        return X

class TransformerEncoder(nn.Module):
    def __init__(self, head_amount=4, embedding_size=128, layer_amount=1, dropout=0.1):
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
    print(input)
    print(X)