import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention gate for Attention U-Net.
    g = decoder gating signal
    x = encoder skip connection
    """
    def __init__(self, gate_channels, skip_channels, inter_channels):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1),
            nn.BatchNorm2d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(AttentionUNet, self).__init__()

        # Encoder
        self.encoder1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder + Attention Gates
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(gate_channels=512, skip_channels=512, inter_channels=256)
        self.decoder4 = DoubleConv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(gate_channels=256, skip_channels=256, inter_channels=128)
        self.decoder3 = DoubleConv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(gate_channels=128, skip_channels=128, inter_channels=64)
        self.decoder2 = DoubleConv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(gate_channels=64, skip_channels=64, inter_channels=32)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder 4
        d4 = self.upconv4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat((e4_att, d4), dim=1)
        d4 = self.decoder4(d4)

        # Decoder 3
        d3 = self.upconv3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat((e3_att, d3), dim=1)
        d3 = self.decoder3(d3)

        # Decoder 2
        d2 = self.upconv2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat((e2_att, d2), dim=1)
        d2 = self.decoder2(d2)

        # Decoder 1
        d1 = self.upconv1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat((e1_att, d1), dim=1)
        d1 = self.decoder1(d1)

        return self.final_conv(d1)


if __name__ == "__main__":
    model = AttentionUNet(in_channels=2, out_channels=1)
    x = torch.randn(2, 2, 256, 256)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)