import torch
import torch.nn as nn


# class Discriminator(nn.Module):
#     def __init__(self, inp_dim=1):
#         super(Discriminator, self).__init__()
#
#
#         self.model = nn.Sequential(
#             nn.Linear(inp_dim, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 128),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(128, 1),
#         )
#
#     def forward(self, img):
#         print(img.shape)
#         # Concatenate label embedding and image to produce input
#         validity = self.model(img)
#         return validity
#

class Discriminator(nn.Module):
    def __init__(self, channel=1, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(

            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),

        )
        self.end = nn.Sequential(

            nn.Linear(1480, 1, bias=True),
            nn.Sigmoid()
        )


    def forward(self, input):

        output = self.main(input)
        #print(output.shape)
        output = output.view(output.size()[0],-1)
        #print(output.shape)

        output = self.end(output)

        #print(output.shape)

        return output.view(-1, 1).squeeze(1)
