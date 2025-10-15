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
    if not mid_channels:
        mid_channels = out_channels
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )


class Down(nn.Module):
    """MaxPool + double_conv"""    
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Upscaling block using ConvTranspose2d + skip connection + double_conv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        in_ch_up = in_channels // 2
        self.up = nn.ConvTranspose2d(in_ch_up, in_ch_up, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels)

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


class PixelShuffle(nn.Module):
    """Helps better SuperResolution restoration."""
    def __init__(self, in_channels, out_channels, upscale: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels*(upscale ** 2), kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor):
        return self.ps(self.conv(x))


class UNet(nn.Module):
    def __init__(self, n_channels=1, o_channels=1, bilinear=True):
        super(UNet, self).__init__()
        # Fix the channel parameters to be integers, not lists
        self.n_channels = n_channels if isinstance(
            n_channels, int) else n_channels[0]
        self.o_channels = o_channels if isinstance(
            o_channels, int) else o_channels[0]
        self.bilinear = bilinear

        ### Revise this part. Layers must be defined in config file. ###
        layers = [16, 32, 64, 128, 256]

        # Encoder path
        self.down_layers = []
        self.down_layers.append(self.double_conv(self.n_channels, layers[0]))
        for n_layer in range(len(layers)):
            self.down_layers.append(self.down(layers[n_layer]), layers[n_layer+1])


        # Decoder path
        
        self.up_seq1 = nn.ConvTranspose2d(layers[4], layers[4], kernel_size=4, stride=2, padding=1)
        self.conv1_0 = self.double_conv(layers[4], layers[3])
        self.conv1_1 = self.double_conv(layers[4], layers[2])  # 128+128 inputs
        
        self.up_seq2 = nn.ConvTranspose2d(layers[2], layers[2], kernel_size=4, stride=2, padding=1)
        self.conv2_0 = self.double_conv(layers[2], layers[2])
        self.conv2 = self.double_conv(layers[3], layers[1])  # 64+64 inputs
        
        # For 4x upscaling, use two sequential Conv2DTranspose layers
        self.up_seq3 = nn.ConvTranspose2d(layers[1], layers[1], kernel_size=4, stride=2, padding=1)
        self.conv3_0 = self.double_conv(layers[1], layers[1])
        self.up_seq3_2 = nn.ConvTranspose2d(layers[1], layers[1], kernel_size=4, stride=2, padding=1)
        self.conv3_0_2 = self.double_conv(layers[1], layers[1])
        self.conv3 = self.double_conv(layers[2], layers[0])  # 32+32 inputs
        
        # Final upscaling layers (for 4x total)
        self.up_seq4 = nn.ConvTranspose2d(layers[0], layers[0], kernel_size=4, stride=2, padding=1)
        self.conv4_0 = self.double_conv(layers[0], layers[0])
        self.up_seq4_2 = nn.ConvTranspose2d(layers[0], layers[0], kernel_size=4, stride=2, padding=1)
        self.conv4_0_2 = self.double_conv(layers[0], layers[0])
        self.conv4 = self.double_conv(layers[1], layers[1])  # 16+16 inputs
        
        # # Convolution components
        # self.conv0 = self.double_conv(layers[4], layers[3])
        # self.conv1 = self.double_conv(layers[4], layers[2])  # 64+64 inputs
        # self.conv2_0 = self.double_conv(layers[2], layers[2])
        # self.conv2 = self.double_conv(layers[3], layers[1])   # 32+32 inputs
        # self.conv3_0 = self.double_conv(layers[1], layers[1])
        # self.conv3 = self.double_conv(layers[2], layers[0])    # 16+16 inputs
        # self.conv4_0 = self.double_conv(layers[0], layers[0])
        # self.conv4 = self.double_conv(layers[1], layers[1])   # 8+8 inputs
        
        # Output layer
        self.outc = nn.Conv2d(layers[1], self.o_channels, kernel_size=5, padding=2)

        # Initialize weights for better convergence
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder in action
        x_input = x.clone()  # keep initial tensor untouched
        decoder_results = []  # stores tensors from every layer of encoder for further use in connections
        # down_layers: inc -> down1 -> down2 -> down3 -> ...
        # 8×H×W -> 16×H/2×W/2 -> 32×H/4×W/4 -> 64×H/8×W/8 -> 128×H/16×W/16
        for element in self.down_layers:
            x_input = element(x_input)  # rewrite x_input to use it in the next layer
            decoder_results.append(x_input)

        # Decoder path with 4x upscaling
        in_ch_up = in_ch_cat // 2
        self.up = nn.ConvTranspose2d(in_ch_up, in_ch_up, kernel_size=2, stride=2)
        self.reduce = nn.Identity()

        self.conv = double_conv(in_ch_cat, out_ch)

        # First upscale (1x -> 2x)
        x = self.up_seq1(decoder_results[-1])              # 128×H/8×W/8
        x = self.conv1_0(x)                 # 64×H/8×W/8
        x = torch.cat([x, decoder_results[-2]], dim=1)     # 128×H/8×W/8
        x = self.conv1_1(x)                 # 32×H/8×W/8

        # Second upscale (2x -> 4x)
        x = self.up_seq2(x)               # 32×H/4×W/4
        x = self.conv2_0(x)               # 32×H/4×W/4
        x = torch.cat([x, decoder_results[-3]], dim=1)     # 64×H/4×W/4
        x = self.conv2(x)                 # 16×H/4×W/4

        # Third upscale (4x -> 8x) - Double upsampling for 4x factor
        x = self.up_seq3(x)               # 16×H/2×W/2
        x = self.conv3_0(x)               # 16×H/2×W/2
        x = self.up_seq3(x)               # 16×H×W
        x = self.conv3_0(x)               # 16×H×W

        # Handle padding if needed for skip connection
        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, x2], dim=1)     # 32×H×W
        x = self.conv3(x)                 # 8×H×W

        # Fourth upscale (8x -> 16x) - Double upsampling for 4x factor
        x = self.up_seq4(x)               # 8×H*2×W*2
        x = self.conv4_0(x)               # 8×H*2×W*2
        x = self.up_seq4(x)               # 8×H*4×W*4
        x = self.conv4_0(x)               # 8×H*4×W*4

        # Handle padding if needed for skip connection
        diffY = x.size()[2] - x1.size()[2]
        diffX = x.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, x1], dim=1)     # 16×H*4×W*4
        x = self.conv4(x)                 # 16×H*4×W*4

        # Final output convolution
        x = self.outc(x)                  # o_channels×H*4×W*4
        return x
