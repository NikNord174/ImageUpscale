import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels=1, o_channels=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels[0]
        self.o_channels = o_channels[0]
        self.bilinear = bilinear
        self.device = 'cuda'
        self.inc = self.double_conv(self.n_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        self.down3 = self.down(256, 512)
        self.down4 = self.down(512, 1024 // (2 if bilinear else 1))

        # self.up_seq1 = self._bilinear(
        #     1024, 512, bilinear)

        # # Bottom of U-Net - smallest feature map 8x8
        # # Upsampling path with skip connections
        # self.up_seq1 = self._bilinear(1024, 512, self.bilinear, scale_factor=4)
        # self.up1 = self.up(1024, 512, self.bilinear, scale_factor=4)
        # self.up_seq2 = self._bilinear(512, 256, self.bilinear, scale_factor=4)
        # self.up2 = self.up(512, 256, self.bilinear, scale_factor=4)
        # self.up_seq3 = self._bilinear(256, 128, self.bilinear, scale_factor=2)
        # self.up3 = self.up(256, 128, self.bilinear, scale_factor=2)
        # self.up_seq4 = self._bilinear(128, 64, self.bilinear, scale_factor=2)
        # self.up4 = self.up(128, 64, self.bilinear, scale_factor=2)

        # self.up_self = self.up



        # Decoder path - create all components at init time
        if bilinear:
            # Upsampling parts
            self.up_seq1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.up_seq2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.up_seq3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up_seq4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
            # Convolution parts
            self.conv1 = self.double_conv(512 + 512, 512)  # 512 from encoder + 512 from upsampled
            self.conv2 = self.double_conv(512 + 256, 256)  # 512 from previous layer + 256 from skip
            self.conv3 = self.double_conv(256 + 128, 128)  # 256 from previous layer + 128 from skip
            self.conv4 = self.double_conv(128 + 64, 64)    # 128 from previous layer + 64 from skip
 
        else:
            # Use transposed convolutions
            self.up_seq1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=4)
            self.up_seq2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=4)
            self.up_seq3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
            self.up_seq4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
            
            # Convolution parts
            self.conv1 = self.double_conv(512 + 512, 512)
            self.conv2 = self.double_conv(512 + 256, 256)
            self.conv3 = self.double_conv(256 + 128, 128)
            self.conv4 = self.double_conv(128 + 64, 64)

        # Output layer
        self.outc = self.out_conv(64, self.o_channels, padding=2)

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
            )#.to(self.device)

    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self.double_conv(in_channels, out_channels)
        )
    
    def _bilinear(self, in_channels, out_channels, bilinear=True, scale_factor=4):
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
            self.conv = double_conv(
                in_channels, out_channels)
        return up_seq

    def scale(self, x1, x2):
        # up_seq, conv = self._bilinear(in_channels, out_channels, bilinear)
        # x5 = self.up_seq(x5)
        # Input is CHW
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]
        # Pad x1 if sizes don't match
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return x, x.size(1)

    def up(self, in_channels, out_channels, bilinear=True, scale_factor=4):
        # up_seq, conv = self._bilinear(in_channels, out_channels, bilinear)
        
        # # Upsample x1
        # x1 = self.up_seq(x1) 

        # # If no skip connection provided, just proceed with x1
        # if x2 is None:
        #     return self.conv(x1)

        # # Input is CHW
        # diffY = x1.size()[2] - x2.size()[2]
        # diffX = x1.size()[3] - x2.size()[3]

        # # Pad x1 if sizes don't match
        # x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # Concatenate along channel dimension
        # x = torch.cat([x2, x1], dim=1)
        # self.in_channels = x.size(1)

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
            in_channels, out_channels, kernel_size=1,
            padding=padding, stride=stride)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)      # 128x128
        x2 = self.down1(x1)   # 64x64
        x3 = self.down2(x2)   # 32x32
        x4 = self.down3(x3)   # 16x16
        x5 = self.down4(x4)   # 8x8

        # Decoder path with skip connections
        # x5 = self.up_seq1(x5)
        # x, in_channels = self.scale(x5, x4)
        # x = self.up(in_channels, 512, scale_factor=4)(x)  # 16x16
        # print('x shape: ', x.shape)

        # x = self.up_seq2(x)
        # x, in_channels = self.scale(x, x3)
        # x = self.up(in_channels, 256, scale_factor=4)(x)  # 32x32
        # print('x shape: ', x.shape)

        # x = self.up_seq3(x)
        # x, in_channels = self.scale(x, x2)
        # x = self.up(in_channels, 128, scale_factor=2)(x)  # 64x64
        # print('x shape: ', x.shape)

        # x = self.up_seq4(x)
        # x, in_channels = self.scale(x, x1)
        # x = self.up(in_channels, 64, scale_factor=2)(x)  # 128x128
        # print('x shape: ', x.shape)


        # First upscale
        x = self.up_seq1(x5)
        
        # Handle size differences for concatenation
        diffY = x.size()[2] - x4.size()[2]
        diffX = x.size()[3] - x4.size()[3]
        x4 = F.pad(x4, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        # Second upscale
        x = self.up_seq2(x)

        diffY = x.size()[2] - x3.size()[2]
        diffX = x.size()[3] - x3.size()[3]
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)


        # Third upscale
        x = self.up_seq3(x)
        diffY = x.size()[2] - x2.size()[2]
        diffX = x.size()[3] - x2.size()[3]
        x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        # Fourth upscale
        x = self.up_seq4(x)
        diffY = x.size()[2] - x1.size()[2]
        diffX = x.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        x = self.outc(x)      # 512x512
        return x
