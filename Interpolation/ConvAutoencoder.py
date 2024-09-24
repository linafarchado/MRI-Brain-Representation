import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, out_channels):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)