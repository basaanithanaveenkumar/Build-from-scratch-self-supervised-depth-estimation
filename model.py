import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Callable

from urllib.request import urlopen
from PIL import Image
import timm
from depth_network import DepthDecoder


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
