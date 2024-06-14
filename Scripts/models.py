import numpy as np
import torch
import torch.nn as nn

class CNN_Transformer(nn.Module):
    def __init__(self):
        super(CNN_Transformer, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64*256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.regression_head(x)
        return x

if __name__ == '__main__':
    model = CNN_Transformer()
    print(model)

    # Input is a batch of 10 segments of 1024 samples each
    x = torch.randn(10, 1, 1024)
    y = model(x)

    print(y)
