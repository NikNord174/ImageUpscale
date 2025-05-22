import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels=1, o_channels=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels[0]
        self.o_channels = o_channels[0]
        self.bilinear = bilinear
        self.inc = self.double_conv(self.n_channels, 8)
        self.down1 = self.down(8, 16)
        self.down2 = self.down(16, 32)
        self.down3 = self.down(32, 64)
        self.down4 = self.down(64, 128)

        # Decoder path - create all components at init time
        # Upsampling parts
        self.up_seq1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up_seq2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up_seq3 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.up_seq4 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        # Convolution parts
        self.conv0 = self.double_conv(128, 64)
        self.conv1 = self.double_conv(64+64, 32)
        self.conv2 = self.double_conv(32+32, 16)
        self.conv2_0 = self.double_conv(32, 32)
        self.conv3 = self.double_conv(16+16, 8)
        self.conv3_0 = self.double_conv(16, 16)
        self.conv4 = self.double_conv(8+8, 16)
        self.conv4_0 = self.double_conv(8, 8)

        # Output layer
        self.outc = self.out_conv(16, self.o_channels, padding=2)

    def double_conv(self, in_channels, out_channels, mid_channels=None):
        if not mid_channels:
            mid_channels = out_channels
        return nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels,
                kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(
                mid_channels, out_channels,
                kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
            )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def _bilinear(
            self, in_channels, out_channels, bilinear=True, scale_factor=4):
        if bilinear:
            up_seq = nn.Upsample(
                scale_factor=scale_factor,
                mode='bilinear', align_corners=True)
            self.conv = self.double_conv(
                in_channels,
                out_channels,
                in_channels // 2)
        else:
            up_seq = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=scale_factor)
            self.conv = self.double_conv(
                in_channels, out_channels)
        return up_seq

    def up(self, in_channels, out_channels, bilinear=True, scale_factor=2):
        # Upsampling layer with selectable method
        if self.bilinear:
            # Adjust channels for bilinear upsampling
            conv = self.double_conv(
                in_channels,
                out_channels,
                in_channels // 2)
        else:
            conv = self.double_conv(
                in_channels, out_channels)

        return conv

    def out_conv(self, in_channels, out_channels, padding=0, stride=1):
        return nn.Conv2d(
            in_channels, out_channels, kernel_size=5,
            padding=padding, stride=stride)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # 8x128x128
        x2 = self.down1(x1)   # 16x64x64
        x3 = self.down2(x2)   # 32x32x32
        x4 = self.down3(x3)   # 64x16x16
        x5 = self.down4(x4)   # 128x8x8

        # First upscale
        x = self.up_seq1(x5)  # 128x16x16
        x = self.conv0(x)  # 64x16x16
        x = torch.cat([x, x4], dim=1)  # 64+64x16x16
        x = self.conv1(x)  # 32x16x16

        # Second upscale
        x = self.up_seq2(x)  # 32x32x32
        x = self.conv2_0(x)  # 32x32x32
        x = torch.cat([x, x3], dim=1)  # 32+32x32x32
        x = self.conv2(x)  # 16x32x32

        # Third upscale
        x = self.up_seq3(x)  # 16x64x64
        x = self.conv3_0(x)  # 16x64x64
        x = self.up_seq3(x)  # 16x64x64
        x = self.conv3_0(x)  # 16x64x64
        # Input is CHW
        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]

        # Pad x1 if sizes don't match
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x2], dim=1)  # 16+16x64x64
        x = self.conv3(x)  # 8x64x64

        # Fourth upscale
        x = self.up_seq4(x)  # 8x128x128
        x = self.conv4_0(x)  # 8x128x128
        x = self.up_seq4(x)  # 8x128x128
        x = self.conv4_0(x)  # 8x128x128
        # Input is CHW
        diffY = x.size()[2] - x1.size()[2]
        diffX = x.size()[3] - x1.size()[3]

        # Pad x1 if sizes don't match
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x1], dim=1)  # 8+8x128x128
        x = self.conv4(x)  # 16x128x128

        x = self.outc(x)  # 1x128x128
        return x
