import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(in_channels, out_channels, mid_channels=None) -> nn.Sequential:
    """Simple block of double Conv layers at one level of U-network.

    Args:
        in_channels (_type_): _description_
        out_channels (_type_): _description_
        mid_channels (_type_, optional): _description_. Defaults to None.

    Returns:
        nn.Sequential: stack of layers for one level of UNet.
    """
    if mid_channels is None:
        mid_channels = out_channels
    return nn.Sequential(
        nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class Down(nn.Module):
    """MaxPool + double_conv"""
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscaling block using ConvTranspose2d + skip connection + double_conv."""
    def __init__(self, in_channels, skip_channels):
        super().__init__()
        # in_ch_up = in_channels // 2
        self.up = nn.ConvTranspose2d(
            in_channels, skip_channels, kernel_size=2, stride=2)
        self.conv = double_conv(skip_channels * 2, skip_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        x = self.up(x)

        # Pad if needed (handles odd dimensions)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class PixelShuffleHead(nn.Module):
    """Helps better SuperResolution restoration."""
    def __init__(self, in_channels, out_channels, upscale: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels*(upscale ** 2), kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor):
        return self.ps(self.conv(x))


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, sr_upscale=4):
        super(UNet, self).__init__()
        # Fix the channel parameters to be integers, not lists
        self.in_channels = in_channels if isinstance(
            in_channels, int) else in_channels[0]
        self.out_channels = out_channels if isinstance(
            out_channels, int) else out_channels[0]
        self.sr_upscale = sr_upscale

        ### Revise this part. Layers must be defined in config file. ###
        self.layers = [16, 32, 64, 128, 256]

        # Encoder path
        self.inc = double_conv(self.in_channels, self.layers[0])
        self.downs = nn.ModuleList()  # creates a properly indexed by torch list of Modules
        for in_layer, out_layer in zip(self.layers[:-1], self.layers[1:]):
            self.downs.append(Down(in_layer, out_layer))

        # Decoder path
        self.ups = nn.ModuleList()
        current_channels = self.layers[-1]
        for skip_channels in self.layers[-2::-1]:  # 128,64,32,16
            self.ups.append(Up(current_channels, skip_channels))
            current_channels = skip_channels

        if self.sr_upscale == 1:
            self.outc = nn.Conv2d(
                current_channels, self.out_channels, kernel_size=1)
            self.sr_head = None
        else:
            self.pre_sr = nn.Sequential(
                nn.Conv2d(
                    current_channels, current_channels, kernel_size=3,
                    padding=1, bias=False),
                nn.BatchNorm2d(current_channels),
                nn.LeakyReLU(inplace=True),
            )
            self.sr_head = PixelShuffleHead(
                current_channels, self.out_channels, upscale=self.sr_upscale)
            self.outc = None

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        x0 = self.inc(x)
        skips = [x0]
        x_i = x0
        for down in self.downs:
            x_i = down(x_i)
            skips.append(x_i)

        # Decoder
        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)

        # Output
        if self.sr_upscale == 1:
            return self.outc(x)
        else:
            x = self.pre_sr(x)
            return self.sr_head(x)


if __name__ == '__main__':
    x = torch.randn(2, 1, 32, 32)
    net = UNet()
    print(net(x).shape)
