import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 2D encoder of 28x28 images 
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 14*14),
            nn.ReLU(True),
            nn.Linear(14*14, 7*7),
            nn.ReLU(True),
            nn.Linear(7*7, 2),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 7*7),
            nn.ReLU(True),
            nn.Linear(7*7, 14*14),
            nn.ReLU(True),
            nn.Linear(14*14, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x.reshape(-1, 28*28))).reshape(-1, 1, 28, 28)
