import torch
from torch.utils.data import DataLoader
import cv2
from dotenv import load_dotenv
import os
from torchvision import transforms
from torch.utils import data
from loguru import logger
import sys

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "video_dataset.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB"
)

# Alternative version that works with video files directly
class VideoPairDataset(data.Dataset):
    """
    Dataloader that extracts frame pairs directly from video files
    """
    
    def __init__(self, video_path, frame_gap=2, img_size=(192, 640), max_frames=1000):
        logger.info(f"Initializing VideoPairDataset with video: {video_path}")
        logger.debug(f"Parameters - frame_gap: {frame_gap}, img_size: {img_size}, max_frames: {max_frames}")
        
        self.video_path = video_path
        self.frame_gap = frame_gap
        self.img_size = img_size
        self.max_frames = max_frames
        
        # Validate video file exists
        if not os.path.exists(self.video_path):
            logger.error(f"Video file not found: {self.video_path}")
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Extract all frames from video
        logger.info("Starting frame extraction from video...")
        self.frames = self._extract_frames()
        logger.success(f"Successfully extracted {len(self.frames)} frames from video")
        
        if len(self.frames) == 0:
            logger.warning("No frames were extracted from the video file")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.debug("Dataset initialization completed")
    
    def _extract_frames(self):
        """Extract frames from video file"""
        logger.debug(f"Opening video file: {self.video_path}")
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {self.video_path}")
            raise IOError(f"Failed to open video file: {self.video_path}")
        
        frames = []
        frame_count = 0
        
        # Get video properties for logging
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video properties - FPS: {fps:.2f}, Total frames: {total_frames}")
        
        while frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                logger.debug(f"Reached end of video at frame {frame_count}")
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
            
            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logger.debug(f"Extracted {frame_count} frames...")
        
        cap.release()
        logger.debug(f"Frame extraction completed. Total frames extracted: {len(frames)}")
        return frames
    
    def __len__(self):
        dataset_length = len(self.frames) - self.frame_gap
        logger.trace(f"Dataset length calculation: {len(self.frames)} frames - {self.frame_gap} gap = {dataset_length}")
        return dataset_length
    
    def __getitem__(self, index):
        logger.trace(f"Accessing dataset item at index: {index}")
        
        if index >= len(self):
            logger.error(f"Index {index} out of bounds for dataset length {len(self)}")
            raise IndexError(f"Index {index} out of bounds")
        
        present_frame = self.frames[index + self.frame_gap]
        past_frame = self.frames[index]
        
        present_tensor = self.transform(present_frame)
        past_tensor = self.transform(past_frame)
        
        sample_data = {
            'present': present_tensor,
            'past': past_tensor,
            'present_idx': index + self.frame_gap,
            'past_idx': index
        }
        
        logger.trace(f"Sample {index}: present_idx={sample_data['present_idx']}, past_idx={sample_data['past_idx']}")
        return sample_data


def test_video_pair_dataset(VIDEO_PATH):
    logger.info("Starting VideoPairDataset test")
    
    try:
        # Initialize the dataset
        logger.info(f"Creating dataset with video: {VIDEO_PATH}")
        dataset = VideoPairDataset(VIDEO_PATH, frame_gap=2, img_size=(192, 640), max_frames=30)

        # Sanity check: dataset length
        logger.info(f"Dataset created successfully. Length: {len(dataset)}")
        
        # Check a few samples
        test_samples = min(3, len(dataset))
        logger.info(f"Testing {test_samples} samples...")
        
        for idx in range(test_samples):
            sample = dataset[idx]
            logger.debug(f"Sample {idx} validation:")
            logger.debug(f"  present idx: {sample['present_idx']}")
            logger.debug(f"  past idx: {sample['past_idx']}")
            logger.debug(f"  present shape: {sample['present'].shape}")
            logger.debug(f"  past shape: {sample['past'].shape}")
            
            # Validate shapes
            if sample['present'].shape != torch.Size([3, 192, 640]):
                logger.error(f"Sample {idx}: Present frame shape mismatch: {sample['present'].shape}")
                raise AssertionError("Shape mismatch")
            
            if sample['past'].shape != torch.Size([3, 192, 640]):
                logger.error(f"Sample {idx}: Past frame shape mismatch: {sample['past'].shape}")
                raise AssertionError("Shape mismatch")
            
            # Validate frame gap
            expected_gap = 2
            if sample['present_idx'] != sample['past_idx'] + expected_gap:
                logger.error(f"Sample {idx}: Frame gap mismatch. Expected {expected_gap}, got {sample['present_idx'] - sample['past_idx']}")
                raise AssertionError("Frame gap mismatch")
            
            logger.success(f"Sample {idx} validation passed")

        # Test DataLoader batching
        logger.info("Testing DataLoader batching...")
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        logger.debug(f"Batch present shape: {batch['present'].shape}")
        logger.debug(f"Batch past shape: {batch['past'].shape}")
        
        # Validate batch shapes
        if batch['present'].shape != torch.Size([2, 3, 192, 640]):
            logger.error(f"Batch present shape incorrect: {batch['present'].shape}")
            raise AssertionError("Batch shape mismatch")
        
        if batch['past'].shape != torch.Size([2, 3, 192, 640]):
            logger.error(f"Batch past shape incorrect: {batch['past'].shape}")
            raise AssertionError("Batch shape mismatch")
        
        logger.success("DataLoader batching test passed")
        logger.success("All tests completed successfully!")
        
        return dataset, loader
        
    except Exception as e:
        logger.exception(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    # Load environment variables
   # Replace with actual path to your test video file
    load_dotenv()
    VIDEO_PATH = os.getenv('video_path')
    
    if not VIDEO_PATH:
        logger.error("video_path environment variable not set or video path not found")
        logger.info("Please set video_path environment variable to your video file path")
        exit(1)
    
    logger.info(f"Using video path from environment: {VIDEO_PATH}")
    
    try:
        dataset, loader = test_video_pair_dataset(VIDEO_PATH)
        logger.success("VideoPairDataset test completed successfully!")
        
    except Exception as e:
        logger.critical(f"VideoPairDataset test failed: {e}")
