import numpy as np
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, encoding_dim),
        #     nn.Tanh(),
        #     nn.Linear(encoding_dim, encoding_dim // 2),
        #     nn.ReLU()
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(encoding_dim // 2, encoding_dim),
        #     nn.Tanh(),
        #     nn.Linear(encoding_dim, input_dim),
        #     nn.ReLU()
        # )

        # The Kitsune AE model
        self.encoder = nn.Sequential(nn.Linear(input_dim, int(input_dim*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(input_dim*0.75), int(input_dim*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(input_dim*0.5),int(input_dim*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(input_dim*0.25),int(input_dim*0.1)))

        self.decoder = nn.Sequential(nn.Linear(int(input_dim*0.1),int(input_dim*0.25)),
                                     nn.ReLU(True),
                                     nn.Linear(int(input_dim*0.25),int(input_dim*0.5)),
                                     nn.ReLU(True),
                                     nn.Linear(int(input_dim*0.5),int(input_dim*0.75)),
                                     nn.ReLU(True),
                                     nn.Linear(int(input_dim*0.75),int(input_dim)),
                                     )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x