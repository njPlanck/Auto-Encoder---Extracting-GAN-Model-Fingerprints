import torch
import torch.nn as nn
import torch.nn.functional as F
from xceptionnet import Xception 

# ----------------- Helper Decoder Class ----------------- #
# ----------------- Helper Decoder Class ----------------- #
class SimpleDecoder(nn.Module):
    """
    Upsampling decoder to reconstruct the image from the bottleneck vector.
    Uses ConvTranspose2d for sharper, learnable upsampling.
    """
    def __init__(self, in_features=4096, out_channels=1):
        super().__init__()
        # Project bottleneck to 4x4 feature map
        # NOTE: If in_features is increased (e.g., from 4096 to 8192), 
        # this FC layer will automatically accommodate the size.
        self.fc = nn.Linear(in_features, 1024 * 4 * 4)
        
        # New decoder blocks using ConvTranspose2d (Learnable Upsampling)
        self.upsample_block = nn.Sequential(
            # 1. 4x4 -> 8x8 (Replaces nn.Upsample + Conv2d)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 2. 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 3. 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 4. 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64x64 → 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final output layer
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            
            # RECOMMENDED: Constrain output to [-1, 1] to match normalized input range
            nn.Tanh() 
        )

    def forward(self, x, target_size):
        B = x.size(0)
        # Reshape bottleneck vector to starting feature map
        x = self.fc(x).view(B, 1024, 4, 4)
        x = self.upsample_block(x)
        
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x

# ----------------- Bottleneck Autoencoder ----------------- #
class BottleneckAutoencoder(nn.Module):
    def __init__(self, bottleneck_features=4096, out_ch=1):
        super().__init__()
        
        # --- Encoder ---
        self.encoder = Xception()
        self.encoder.conv1 = nn.Conv2d(
            in_channels=1, # <--- CRITICAL FIX: Change from 3 to 1
            out_channels=32, 
            kernel_size=3, 
            stride=2, 
            padding=1, 
            bias=False
            )
        self.encoder.classifier = nn.Identity()  # remove classifier

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 10x10 -> 1x1
            nn.Flatten(),
            nn.Linear(2048, bottleneck_features),  # match decoder input
            nn.ReLU(inplace=True)
        )

        # --- Decoder ---
        self.decoder = SimpleDecoder(in_features=bottleneck_features, out_channels=out_ch)
        
    def forward(self, x):
        B, C, H, W = x.size()
        features = self.encoder(x)           # B x 2048 x H' x W'
        l_vector = self.bottleneck(features) # B x bottleneck_features
        reconstructed = self.decoder(l_vector, target_size=(H, W))
        return reconstructed
