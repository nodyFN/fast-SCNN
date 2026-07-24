import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """UNet model for Knowledge Distillation teacher."""
    def __init__(self, in_channels: int = 3, out_channels: int = 1, init_features: int = 32) -> None:
        super().__init__()
        features = init_features
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.bottleneck = DoubleConv(features * 8, features * 16)

        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features * 16, features * 8, 1, bias=False)
        )
        self.decoder4 = DoubleConv(features * 16, features * 8)

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features * 8, features * 4, 1, bias=False)
        )
        self.decoder3 = DoubleConv(features * 8, features * 4)

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features * 4, features * 2, 1, bias=False)
        )
        self.decoder2 = DoubleConv(features * 4, features * 2)

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features * 2, features, 1, bias=False)
        )
        self.decoder1 = DoubleConv(features * 2, features)

        self.conv = nn.Conv2d(features, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        if dec4.shape != enc4.shape:
            dec4 = F.interpolate(dec4, size=enc4.shape[2:], mode="bilinear", align_corners=False)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        if dec3.shape != enc3.shape:
            dec3 = F.interpolate(dec3, size=enc3.shape[2:], mode="bilinear", align_corners=False)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        if dec2.shape != enc2.shape:
            dec2 = F.interpolate(dec2, size=enc2.shape[2:], mode="bilinear", align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        if dec1.shape != enc1.shape:
            dec1 = F.interpolate(dec1, size=enc1.shape[2:], mode="bilinear", align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)
