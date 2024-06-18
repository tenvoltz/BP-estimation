import numpy as np
import torch
import torch.nn as nn
from transformer import TransformerEncoder

class CNN_Transformer(nn.Module):
    def __init__(self):
        super(CNN_Transformer, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # Make this 2 later on
            nn.MaxPool1d(kernel_size=8, stride=8)
        )
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # Make this 2 later on
            nn.MaxPool1d(kernel_size=8, stride=8)
        )
        self.transformer = TransformerEncoder(
            head_amount=8,
            embedding_size=64 * 16,
            layer_amount=6,
            dropout=0.1
        )
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 16, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten the output for the transformer
        x = x.view(x.size(0), -1)
        # No embedding for the feature yet, so we add a dummy dimension
        # Essentially, the input is raw feature
        x = x.unsqueeze(2)
        x = self.transformer(x)
        # Get the last transformer output
        x = x[:, -1, :]
        x = self.regression_head(x)
        return x

if __name__ == '__main__':
    model = CNN_Transformer()

    # Input is a batch of 10 segments of 1 channel of 1024 samples
    x = torch.randn(10, 1, 1024)
    y = model(x)

    print(y)
