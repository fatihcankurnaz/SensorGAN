import torch
import torch.nn as nn




class Discriminator(nn.Module):
    def __init__(self, in_channels=5, ngpu=2):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels , 16, normalization=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 1, 2, padding=1, bias=False),

        )
        self.final = nn.Sequential(
            nn.Linear(1975, 1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        # Concatenate image and condition image by channels to produce input


        output = self.model(inp)

        output = output.view(output.size()[0],-1)
        #print(output.shape)

        output = self.final(output)

        #print(output.shape)

        return output.view(-1, 1).squeeze(1)


class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, ngpu=2):
        super(PixelDiscriminator, self).__init__()
        self.ngpu = ngpu
        def discriminator_block(in_filters, out_filters, normalization=True, dropout=0.0):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1, bias=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels + out_channels , 64, normalization=False, dropout=0.5),
            *discriminator_block(64, 128, dropout=0.5),
            *discriminator_block(128, 256, dropout=0.5),
            *discriminator_block(256, 512, dropout=0.5),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False),

        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)

        #print(output.shape)

        return self.model(img_input)

