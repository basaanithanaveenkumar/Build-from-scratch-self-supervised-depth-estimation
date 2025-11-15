import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable

from urllib.request import urlopen
from PIL import Image
import timm


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


def load_model_and_transforms(device: torch.device) -> Tuple[nn.Module, Callable]:
    """
    Loads pretrained DINOv3 backbone and prepares input transforms.

    Args:
        device (torch.device): Device to load the model on.

    Returns:
        model (nn.Module): ViT backbone model in features_only mode (eval).
        transforms (Callable): Input preprocessing transforms.
    """
    model = timm.create_model(
        'vit_huge_plus_patch16_dinov3_qkvb.lvd1689m',
        pretrained=True,
        features_only=True,
    )
    model.eval()
    model.to(device)

    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    return model, transforms


def infer_depth(img: Image.Image, model: nn.Module, depth_decoder: nn.Module,
                transforms: Callable, device: torch.device) -> torch.Tensor:
    """
    Run inference to get depth map from input PIL image.

    Args:
        img (Image.Image): Input RGB image.
        model (nn.Module): Backbone model.
        depth_decoder (nn.Module): Depth decoder.
        transforms (Callable): Preprocessing transforms.
        device (torch.device): Device to run inference on.

    Returns:
        torch.Tensor: Depth map tensor (B, 1, H, W) on CPU.
    """
    input_tensor = transforms(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
        depth_decoder.eval()
        depth_map = depth_decoder(features).cpu()
    return depth_map


def test_depth_decoder():
    """Run example inference pipeline and verify output shapes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, transforms = load_model_and_transforms(device)

    url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    img = Image.open(urlopen(url)).convert('RGB')

    # Extract features once
    with torch.no_grad():
        features = model(transforms(img).unsqueeze(0).to(device))

    feature_dims = [f.shape[1] for f in features]
    depth_decoder = DepthDecoder(feature_dims).to(device)

    depth_map = infer_depth(img, model, depth_decoder, transforms, device)

    # Print shapes for verification
    print("Feature shapes:")
    for i, f in enumerate(features):
        print(f"  Feature {i}: {f.shape}")
    print(f"Depth map shape: {depth_map.shape}")


if __name__ == "__main__":
    test_depth_decoder()
