import torch
import torch.nn as nn

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # 2D encoder of 28x28 images 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.latent = nn.Linear(8 * 2 * 2, 2)

        self.latent2 = nn.Linear(2, 8 * 2 * 2)

        # 2D decoder to 28x28
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        latent = self.latent(encoded.view(-1, 8 * 2 * 2))
        latent2 = self.latent2(torch.sigmoid(latent)).view(-1, 8, 2, 2)
        decoded = self.decoder(latent2)
        return decoded
