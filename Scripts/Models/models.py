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

SIGNAL_LENGTH = int(os.getenv('INPUT_LENGTH'))
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
            nn.LazyConv1d(64 * 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(64 * 1),
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
        self.pool3by3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv1d(64 * 1, kernel_size=1),
            nn.BatchNorm1d(64 * 1),
            nn.ReLU(),
        )
        self.conv1by1combine = nn.Sequential(
            nn.LazyConv1d(64 * 4, kernel_size=1),
            nn.BatchNorm1d(64 * 4),
            nn.ReLU(),
            #nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.transformer = TransformerEncoder(
            head_amount=4,
            embedding_size=64 * 4,
            layer_amount=1,
            dropout=0.1
        )
        self.GlobalAvgPool = nn.AvgPool1d(kernel_size=64 * 4)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.LazyLinear(2),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.conv1by1(x)
        x2 = self.conv3by3(x)
        x3 = self.conv5by5(x)
        x4 = self.pool3by3(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv1by1combine(x)
        x = x.permute(0, 2, 1)
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

class Inception_Module(nn.Module):
    def __init__(self, out_channels, shrink_factor=4):
        super(Inception_Module, self).__init__()
        self.conv1by1 = nn.Sequential(
            nn.LazyConv1d(out_channels // shrink_factor, kernel_size=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
        )
        self.conv3by3 = nn.Sequential(
            nn.LazyConv1d(out_channels // shrink_factor, kernel_size=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
            nn.Conv1d(out_channels // shrink_factor, out_channels // shrink_factor, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
        )
        self.conv5by5 = nn.Sequential(
            nn.LazyConv1d(out_channels // shrink_factor, kernel_size=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
            nn.Conv1d(out_channels // shrink_factor, out_channels // shrink_factor, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
            nn.Conv1d(out_channels // shrink_factor, out_channels // shrink_factor, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
        )
        self.pool3by3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.LazyConv1d(out_channels // shrink_factor, kernel_size=1),
            nn.BatchNorm1d(out_channels // shrink_factor),
            nn.ReLU(),
        )
        self.conv1by1combine = nn.Sequential(
            nn.Conv1d(out_channels // shrink_factor * 4, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.conv1by1(x)
        x2 = self.conv3by3(x)
        x3 = self.conv5by5(x)
        x4 = self.pool3by3(x)
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv1by1combine(x)

        return x

class Transformer_Only(nn.Module):
    def __init__(self, bias_init=None, alpha=0.5):
        super(Transformer_Only, self).__init__()
        self.alpha = alpha
        self.embedding = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.transformer = nn.Sequential(
            TransformerEncoder(
                head_amount=4,
                embedding_size=64,
                layer_amount=1,
                dropout=0.1
            ),
            nn.AvgPool1d(kernel_size=64),
            nn.Flatten(),
        )
        self.conv = nn.Sequential(
            Inception_Module(128, shrink_factor=16),
            nn.AvgPool1d(kernel_size=1024),
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
        )
        """
        self.demographics_embedding = nn.Sequential(
            #nn.LazyLinear(256),
            #nn.BatchNorm1d(256),
            #nn.ReLU(),
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        """
        BP_regressor = nn.Linear(512,2)
        if bias_init is not None:
            with torch.no_grad():
                BP_regressor.bias.copy_(bias_init)
        self.regression_head = nn.Sequential(
            #nn.Dropout(0.1),
            #nn.LazyLinear(64),
            #nn.BatchNorm1d(64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            BP_regressor,
            nn.ReLU()
        )

    def forward(self, x):
        # demographics = x['demographics']
        x_conv = self.conv(x['signals'])
        x = self.embedding(x['signals'])
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        # demographics = self.demographics_embedding(demographics)
        # Element-wise addition of x and demographics
        # x = x + demographics
        # Element-wise addition of x and conv scaled by alpha
        #  x = x * self.alpha + x_conv * (1 - self.alpha)# kinda bad
        x = x + x_conv
        x = self.regression_head(x)
        return x

class Dummy_Model(nn.Module):
    def __init__(self):
        super(Dummy_Model, self).__init__()
        self.regresion_head = nn.Sequential(
            nn.LazyLinear(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = x['signals']
        x = x.view(x.size(0), -1)
        x = self.regresion_head(x)
        return x

if __name__ == '__main__':
    model = Dummy_Model()
    x = {
        'signals': torch.FloatTensor(torch.randn(1, 3, SIGNAL_LENGTH)),
        # 'demographics': torch.FloatTensor(torch.randn(BATCH_SIZE, 3))
    }
    with torch.no_grad():
        y = model(x)
        print(y)
        torchinfo.summary(model, input_data=[x], verbose=1)
