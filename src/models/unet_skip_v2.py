import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels=1, o_channels=1, bilinear=True):
        super(UNet, self).__init__()
        # Fix the channel parameters to be integers, not lists
        self.n_channels = n_channels if isinstance(n_channels, int) else n_channels[0]
        self.o_channels = o_channels if isinstance(o_channels, int) else o_channels[0]
        self.bilinear = bilinear

        # layers = [8, 16, 32, 64, 128]
        layers = [16, 32, 64, 128, 256]
        
        # Encoder path
        self.inc = self.double_conv(self.n_channels, layers[0])
        self.down1 = self.down(layers[0], layers[1])
        self.down2 = self.down(layers[1], layers[2])
        self.down3 = self.down(layers[2], layers[3])
        self.down4 = self.down(layers[3], layers[4])
        
        # Decoder path for 4x upscaling
        # Upsampling components
        # self.up_seq1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_seq2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_seq3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up_seq4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up_seq1 = nn.ConvTranspose2d(layers[4], layers[4], kernel_size=4, stride=2, padding=1)
        self.conv0 = self.double_conv(layers[4], layers[3])
        self.conv1 = self.double_conv(layers[4], layers[2])  # 128+128 inputs
        
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
    
    def double_conv(self, in_channels, out_channels, mid_channels=None):
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
    
    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )
    
    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # 8×H×W
        x2 = self.down1(x1)   # 16×H/2×W/2
        x3 = self.down2(x2)   # 32×H/4×W/4
        x4 = self.down3(x3)   # 64×H/8×W/8
        x5 = self.down4(x4)   # 128×H/16×W/16
        
        # Decoder path with 4x upscaling
        # First upscale (1x -> 2x)
        x = self.up_seq1(x5)              # 128×H/8×W/8
        x = self.conv0(x)                 # 64×H/8×W/8
        x = torch.cat([x, x4], dim=1)     # 128×H/8×W/8
        x = self.conv1(x)                 # 32×H/8×W/8
        
        # Second upscale (2x -> 4x)
        x = self.up_seq2(x)               # 32×H/4×W/4
        x = self.conv2_0(x)               # 32×H/4×W/4
        x = torch.cat([x, x3], dim=1)     # 64×H/4×W/4
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
