import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)

class ResNetDNN(nn.Module):
    def __init__(self, input_dim=31, output_dim=100, hidden_dim=1024, num_blocks=8, dropout=0.05):
        super(ResNetDNN, self).__init__()
        
        # 1. Input Layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # 2. Deep Residual Body
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        # 3. Output Layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x