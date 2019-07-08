import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, inp_dim):
        super(Discriminator, self).__init__()


        self.model = nn.Sequential(
            nn.Linear(inp_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity
