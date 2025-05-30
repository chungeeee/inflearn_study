import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # in_channels: decoder에서 오는 채널 수
        # out_channels: 최종적으로 원하는 채널 수
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.reduce_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.reduce_channels = nn.Identity()
        self.conv = DoubleConv(out_channels * 2, out_channels)  # concat 후 채널 수

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.reduce_channels(x1)
        # 크기 맞추기
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNET_ISBI_2012(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.dropout = nn.Dropout(0.5)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.inc(x)         # [B, 64, 512, 512]
        c2 = self.down1(c1)      # [B, 128, 256, 256]
        c3 = self.down2(c2)      # [B, 256, 128, 128]
        c4 = self.down3(c3)      # [B, 512, 64, 64]
        c4 = self.dropout(c4)
        c5 = self.down4(c4)      # [B, 1024, 32, 32]
        c5 = self.dropout(c5)

        # Decoder
        u6 = self.up1(c5, c4)    # [B, 512, 64, 64]
        u7 = self.up2(u6, c3)    # [B, 256, 128, 128]
        u8 = self.up3(u7, c2)    # [B, 128, 256, 256]
        u9 = self.up4(u8, c1)    # [B, 64, 512, 512]
        out = self.outc(u9)      # [B, num_classes, 512, 512]
        return out
    
class UNET_OXFORD_IIIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.conv_last1 = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.conv_last2 = nn.Conv2d(2, num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.inc(x)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        d4 = self.dropout4(c4)
        c5 = self.down4(d4)
        d5 = self.dropout5(c5)
        u6 = self.up1(d5, d4)
        u7 = self.up2(u6, c3)
        u8 = self.up3(u7, c2)
        u9 = self.up4(u8, c1)
        conv9 = self.conv_last1(u9)
        out = self.conv_last2(conv9)
        return out