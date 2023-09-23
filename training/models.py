import torch
import torch.nn as nn

class ConvMaxPoolReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvMaxPoolReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        return x

class ScoreRecognizer(nn.Module):
    def __init__(self, num_classes: int):
        super(ScoreRecognizer, self).__init__()
        self.num_classes = num_classes
        self.layers = nn.Sequential(
            ConvMaxPoolReLU(3, 16),
            ConvMaxPoolReLU(16, 32),
            ConvMaxPoolReLU(32, 64),
        )
        self.linear = nn.Linear(64 * 8 * 8, 256)
        self.classifier = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.linear(x))
        x = self.dropout(x)
        x = self.classifier(x)
        return x