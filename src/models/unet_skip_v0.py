import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block: (Conv2d -> BatchNorm -> ReLU) × 2
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
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

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block with maxpool followed by double convolution
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block with upsampling followed by double convolution
    """
    def __init__(
            self, in_channels, out_channels,
            bilinear=True, scale_factor=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.scale_factor = scale_factor

    def forward(self, x1, x2=None):
        if self.bilinear:
            self.up = nn.Upsample(
                scale_factor=self.scale_factor,
                mode='bilinear', align_corners=True)
            self.conv = DoubleConv(
                self.in_channels,
                self.out_channels,
                self.in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                self.in_channels, self.in_channels // 2,
                kernel_size=2, stride=self.scale_factor)
            self.conv = DoubleConv(
                self.in_channels, self.out_channels)
        # Upsample x1
        x1 = self.up(x1)

        # If no skip connection provided, just proceed with x1
        if x2 is None:
            return self.conv(x1)

        # Input is CHW
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        # Pad x1 if sizes don't match
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        self.in_channels = x.size(1)

        # Upsampling layer with selectable method
        if self.bilinear:
            # Adjust channels for bilinear upsampling
            self.conv = DoubleConv(
                self.in_channels,
                self.out_channels,
                self.in_channels // 2)
        else:
            self.conv = DoubleConv(
                self.in_channels, self.out_channels)

        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution block
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture for 1×128×128 input to 1×512×512 output
    """
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial input processing
        self.inc = DoubleConv(n_channels, 64)

        # Downsampling path
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // (2 if bilinear else 1))

        # Bottom of U-Net - smallest feature map 8x8
        # Upsampling path with skip connections
        self.up1 = Up(1024, 512, bilinear, scale_factor=4)  # 32x32
        self.up2 = Up(512, 256, bilinear, scale_factor=4)  # 128x128
        self.up3 = Up(256, 128, bilinear, scale_factor=2)  # 256x256
        self.up4 = Up(128, 64, bilinear, scale_factor=2)  # 512x512

        # Output layer
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # 128x128
        x2 = self.down1(x1)   # 64x64
        x3 = self.down2(x2)   # 32x32
        x4 = self.down3(x3)   # 16x16
        x5 = self.down4(x4)   # 8x8

        # Decoder path with skip connections
        x = self.up1(x5, x4)  # 16x16
        x = self.up2(x, x3)   # 32x32
        x = self.up3(x, x2)   # 64x64
        x = self.up4(x, x1)   # 128x128

        x = self.outc(x)      # 512x512
        return x
