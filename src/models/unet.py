import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """U-Net that upscales its input 4x.

    The encoder halves the resolution four times; the first two decoder
    stages upsample 4x instead of 2x, so the network comes out two
    doublings ahead of where it started: a 32x32 pattern goes in, a
    128x128 pattern comes out. Skip tensors are zero-padded up to the
    decoder resolution before concatenation.
    """

    def __init__(self, n_channels=1, o_channels=1):
        super().__init__()
        self.n_channels = n_channels
        self.o_channels = o_channels

        self.inc = self.double_conv(n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 512)

        self.up_seq1 = self.upsample(scale_factor=4)
        self.up_seq2 = self.upsample(scale_factor=4)
        self.up_seq3 = self.upsample(scale_factor=2)
        self.up_seq4 = self.upsample(scale_factor=2)

        self.conv1 = self.double_conv(512 + 512, 512)
        self.conv2 = self.double_conv(512 + 256, 256)
        self.conv3 = self.double_conv(256 + 128, 128)
        self.conv4 = self.double_conv(128 + 64, 64)

        self.outc = nn.Conv2d(64, o_channels, kernel_size=1)

    @staticmethod
    def double_conv(in_channels, out_channels, mid_channels=None):
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
            nn.LeakyReLU(inplace=True),
        )

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels),
        )

    @staticmethod
    def upsample(scale_factor):
        return nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear', align_corners=True)

    @staticmethod
    def up_block(x, skip, conv):
        diff_y = x.size(2) - skip.size(2)
        diff_x = x.size(3) - skip.size(3)
        skip = F.pad(skip, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])
        return conv(torch.cat([skip, x], dim=1))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up_block(self.up_seq1(x5), x4, self.conv1)
        x = self.up_block(self.up_seq2(x), x3, self.conv2)
        x = self.up_block(self.up_seq3(x), x2, self.conv3)
        x = self.up_block(self.up_seq4(x), x1, self.conv4)
        return self.outc(x)
