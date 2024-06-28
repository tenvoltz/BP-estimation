import numpy as np
import torch
import torch.nn as nn
import torchinfo
from dotenv import load_dotenv
import os

from Models.transformer import TransformerEncoder

load_dotenv()
DATA_PATH = os.getenv('DATA_PATH')
DATASET_NAME = os.getenv('DATASET_NAME')
load_dotenv(os.path.join(DATA_PATH, DATASET_NAME, '.env'))

SIGNAL_LENGTH = int(os.getenv('SAMPLES_PER_SEGMENT'))
SIGNAL_AMOUNT = int(os.getenv('SIGNAL_AMOUNT'))
BATCH_SIZE = int(os.getenv('BATCH_SIZE'))

class Inception_Transformer(nn.Module):
    def __init__(self):
        super(Inception_Transformer, self).__init__()
        self.conv1by1 = nn.Sequential(
            nn.LazyConv1d(64 * 1, kernel_size=1),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3by3 = nn.Sequential(
            nn.LazyConv1d(64 * 1, kernel_size=1),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
            nn.LazyConv1d(64 * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(64 * 2),
            nn.ReLU(),
            #nn.LazyConv1d(64 * 3, kernel_size=1),
            #nn.BatchNorm1d(64 * 3),
            #nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv5by5 = nn.Sequential(
            nn.LazyConv1d(64 * 1, kernel_size=1),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
            nn.LazyConv1d(64 * 1, kernel_size=5, padding=2),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
            #nn.LazyConv1d(64 * 3, kernel_size=1),
            #nn.BatchNorm1d(64 * 3),
            #nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv1by1combine = nn.Sequential(
            nn.LazyConv1d(64 * 4, kernel_size=1),
            nn.BatchNorm1d(64 * 4),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.transformer = TransformerEncoder(
            head_amount=4,
            embedding_size=SIGNAL_LENGTH,
            layer_amount=1,
            dropout=0.1
        )
        self.GlobalAvgPool = nn.AvgPool1d(kernel_size=SIGNAL_LENGTH)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(2),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1by1(x)
        x2 = self.conv3by3(x)
        x3 = self.conv5by5(x)
        x = torch.cat((x1, x2, x3), 1)
        x = self.conv1by1combine(x)
        x = self.transformer(x)
        x = self.GlobalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)

        return x

class Dense_Transformer(nn.Module):
    def __init__(self):
        super(Dense_Transformer, self).__init__()
        self.embedding = nn.Sequential(
            nn.LazyLinear(1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(1024),
        )
        self.transformer = TransformerEncoder(
            head_amount=8,
            embedding_size=256,
            layer_amount=2,
            dropout=0.1
        )
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(2),
            nn.ReLU()
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = x.view(x.size(0), -1, 256)
        x = self.transformer(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)

        return x

class CNN_Transformer(nn.Module):
    def __init__(self):
        super(CNN_Transformer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.LazyConv1d(32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.transformer = TransformerEncoder(
            head_amount=8,
            embedding_size=SIGNAL_LENGTH // 4,
            layer_amount=2,
            dropout=0.1
        )
        self.GlobalAvgPool = nn.AvgPool1d(kernel_size=SIGNAL_LENGTH // 4)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transformer(x)
        x = self.GlobalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)

        return x

class CNN_Only(nn.Module):
    def __init__(self):
        super(CNN_Only, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)

        return x

class Transformer_Only(nn.Module):
    def __init__(self):
        super(Transformer_Only, self).__init__()
        self.embedding = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.transformer = TransformerEncoder(
            head_amount=4,
            embedding_size=64,
            layer_amount=1,
            dropout=0.1
        )
        self.GlobalAvgPool = nn.AvgPool1d(kernel_size=64)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(2),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = self.GlobalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)

        return x

class Transformer_Only_Classifier(nn.Module):
    def __init__(self):
        super(Transformer_Only_Classifier, self).__init__()
        self.embedding = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.transformer = TransformerEncoder(
            head_amount=4,
            embedding_size=64,
            layer_amount=1,
            dropout=0.1
        )
        self.GlobalAvgPool = nn.AvgPool1d(kernel_size=64)
        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.LazyLinear(1),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = self.GlobalAvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.classification_head(x)

        return x


if __name__ == '__main__':
    model = Inception_Transformer()
    # Input is a batch of 10 segments of 1 channel of 1024 samples
    x = torch.randn(10, SIGNAL_AMOUNT, SIGNAL_LENGTH)
    y = model(x)
    print(y)

    torchinfo.summary(model, (BATCH_SIZE, SIGNAL_AMOUNT, SIGNAL_LENGTH))