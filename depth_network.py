
class DepthDecoder(nn.Module):
    def __init__(self, feature_dims: List[int]):
        """
        Depth decoder that upsamples and fuses multi-scale features into a dense depth map.

        Args:
            feature_dims (List[int]): Number of channels for each feature map from backbone.
        """
        super().__init__()
        if len(feature_dims) != 3:
            raise ValueError("Expecting exactly three feature maps")

        # Conv layers to unify feature channels
        self.conv1 = nn.Conv2d(feature_dims[0], 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_dims[1], 512, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_dims[2], 512, kernel_size=3, padding=1)

        # Transposed convolutions for upsampling
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass to produce depth map.

        Args:
            features (List[torch.Tensor]): List of multi-scale feature maps from backbone.

        Returns:
            torch.Tensor: Predicted depth map (B, 1, H, W)
        """
        f1, f2, f3 = features

        x1 = F.relu(self.conv1(f1))
        x2 = F.relu(self.conv2(f2))
        x3 = F.relu(self.conv3(f3))

        # Fuse features by element-wise addition (consider concat + conv alternative)
        x = x1 + x2 + x3

        # Upsample progressively
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))

        depth = self.final_conv(x)
        return depth