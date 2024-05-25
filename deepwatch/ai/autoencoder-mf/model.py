from torch import nn
import numpy as np

# Positional Encoding Function
def positional_encoding(identifier, dimension):
    position = np.arange(len(identifier)).reshape(-1, 1)
    div_term = np.exp(np.arange(0, dimension, 2) * -(np.log(10000.0) / dimension))
    pos_enc = np.zeros((len(identifier), dimension))
    pos_enc[:, 0::2] = np.sin(position * div_term)
    pos_enc[:, 1::2] = np.cos(position * div_term)
    return pos_enc

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Tanh(),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.Tanh(),
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x