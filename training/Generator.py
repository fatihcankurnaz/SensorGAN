# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F

#
# class double_conv(nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#
#     def __init__(self, in_ch, out_ch):
#         super(double_conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
#
# class inconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(inconv, self).__init__()
#         self.conv = double_conv(in_ch, out_ch)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
#
# class down(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(down, self).__init__()
#         self.mpconv = nn.Sequential(
#             nn.MaxPool2d(2),
#             double_conv(in_ch, out_ch)
#         )
#
#     def forward(self, x):
#         x = self.mpconv(x)
#         return x
#
#
# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=True):
#         super(up, self).__init__()
#
#         #  would be a nice idea if the upsampling could be learned too,
#         #  but my machine do not have enough memory to handle all those weights
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
#
#         self.conv = double_conv(in_ch, out_ch)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#
#         x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2))
#
#         # for padding issues, see
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#
#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x
#
#
# class outconv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(outconv, self).__init__()
#
#         self.conv = nn.Conv2d(in_ch, out_ch, 1)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
#
#
# # full assembly of the sub-parts to form the complete net
#
# import torch.nn.functional as F
#
#
#
# class Generator(nn.Module):
#     def __init__(self, n_channels=1, n_classes=1, ngpu=2):
#         super(Generator, self).__init__()
#         self.ngpu = 2
#         self.inc = inconv(n_channels, 16)
#         self.down1 = down(16, 32)
#         self.down2 = down(32, 64)
#         self.down3 = down(64, 128)
#         self.down4 = down(128, 128)
#         self.up1 = up(256, 64)
#         self.up2 = up(128, 32)
#         self.up3 = up(64, 16)
#         self.up4 = up(32, 16)
#         self.outc = outconv(16, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#
#         x = self.outc(x)
#
#         return x






def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        diffY = skip_input.size()[2] - x.size()[2]
        diffX = skip_input.size()[3] - x.size()[3]

        x = F.pad(x, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat((x, skip_input), 1)

        return x


class Generator(nn.Module):
    def __init__(self, in_channels=5, out_channels=5, ngpu = 2):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.down1 = UNetDown(in_channels, 8, normalize=False)
        self.down2 = UNetDown(8, 16)
        self.down3 = UNetDown(16, 32)
        self.down4 = UNetDown(32, 64, dropout=0.5)
        self.down5 = UNetDown(64, 64, dropout=0.5)
        # self.down6 = UNetDown(64, 64, dropout=0.5)
        # self.down7 = UNetDown(64, 64, dropout=0.5)
        self.down6 = UNetDown(64, 64, normalize=False, dropout=0.5)

        self.up1 = UNetUp(64, 64, dropout=0.5)
        # self.up2 = UNetUp(128, 128, dropout=0.5)
        # self.up3 = UNetUp(64, 64, dropout=0.5)
        self.up4 = UNetUp(128, 64, dropout=0.5)
        self.up5 = UNetUp(128, 32)
        self.up6 = UNetUp(64, 16)
        self.up7 = UNetUp(32, 8)

        self.final = nn.Sequential(
            nn.Upsample(size=(375,1242)),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(16, out_channels, 4, padding=1),
            # nn.Softmax2d(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        u1 = self.up1(d6, d5)
        # u2 = self.up2(u1, d6)
        # u3 = self.up3(u2, d5)
        u4 = self.up4(u1, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)
