import torch
import torch.utils.data as data
import os
from pathlib import Path
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

class SimpleFramePairDataset(data.Dataset):
    """
    Simple dataloader that returns pairs of (present_frame, past_frame)
    for self-supervised depth estimation
    """
    
    def __init__(self, data_path, frame_gap=1, img_size=(192, 640), is_train=True):
        """
        Args:
            data_path: Path to directory containing video frames or videos
            frame_gap: Number of frames between present and past frame
            img_size: (height, width) for resizing
            is_train: Whether this is for training or validation
        """
        self.data_path = Path(data_path)
        self.frame_gap = frame_gap
        self.img_size = img_size
        self.is_train = is_train
        
        # Collect all frame paths
        self.frame_paths = self._collect_frames()
        print(f"Found {len(self.frame_paths)} frames")
        
        # Create transformations
        self.transform = self._create_transforms()
    
    def _collect_frames(self):
        """Collect all frame file paths"""
        frame_paths = []
        
        # If directory contains image files directly
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        for ext in image_extensions:
            frame_paths.extend(self.data_path.glob(f'*{ext}'))
        
        # If no images found, check for subdirectories
        if not frame_paths:
            for subdir in self.data_path.iterdir():
                if subdir.is_dir():
                    for ext in image_extensions:
                        frame_paths.extend(subdir.glob(f'*{ext}'))
        
        # Sort to maintain temporal order
        frame_paths = sorted(frame_paths)
        return frame_paths
    
    def _create_transforms(self):
        """Create image transformations"""
        if self.is_train:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def _load_image(self, image_path):
        """Load and transform single image"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)
    
    def __len__(self):
        # We need at least frame_gap + 1 frames to form a pair
        return len(self.frame_paths) - self.frame_gap
    
    def __getitem__(self, index):
        """
        Returns:
            present_frame: Frame at current time t
            past_frame: Frame at time t - frame_gap
        """
        # Present frame index
        present_idx = index + self.frame_gap
        # Past frame index
        past_idx = index
        
        present_path = self.frame_paths[present_idx]
        past_path = self.frame_paths[past_idx]
        
        # Load frames
        present_frame = self._load_image(present_path)
        past_frame = self._load_image(past_path)
        
        return {
            'present': present_frame,      # Frame at time t
            'past': past_frame,            # Frame at time t-1 (or t-frame_gap)
            'present_path': str(present_path),
            'past_path': str(past_path),
            'present_idx': present_idx,
            'past_idx': past_idx
        }





# Example usage and testing
if __name__ == "__main__":
    # Test with image directory
    dataset = SimpleFramePairDataset(
        data_path='~/hautonomy_gt/data/rgb',
        frame_gap=1,  # Use consecutive frames
        img_size=(192, 640),
        is_train=True
    )
    
    # Create dataloader
    dataloader = data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    # Test one batch
    for batch in dataloader:
        present = batch['present']  # Shape: [batch_size, 3, H, W]
        past = batch['past']        # Shape: [batch_size, 3, H, W]
        
        print(f"Batch - Present: {present.shape}, Past: {past.shape}")
        print(f"Present indices: {batch['present_idx']}")
        print(f"Past indices: {batch['past_idx']}")
        
        # For self-supervised depth estimation:
        # - Use 'present' as target frame (I_t)
        # - Use 'past' as context frame (I_{t-1})
        # The network will learn to predict depth from 'present' and pose between frames
        
        break
    
    print("Dataloader test successful!")