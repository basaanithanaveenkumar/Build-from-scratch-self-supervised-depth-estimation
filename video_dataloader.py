import torch
from torch.utils.data import DataLoader
import cv2
from dotenv import load_dotenv
# Alternative version that works with video files directly
class VideoPairDataset(data.Dataset):
    """
    Dataloader that extracts frame pairs directly from video files
    """
    
    def __init__(self, video_path, frame_gap=2, img_size=(192, 640), max_frames=1000):
        self.video_path = video_path
        self.frame_gap = frame_gap
        self.img_size = img_size
        self.max_frames = max_frames
        
        # Extract all frames from video
        self.frames = self._extract_frames()
        print(f"Extracted {len(self.frames)} frames from video")
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _extract_frames(self):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(str(self.video_path))
        frames = []
        
        frame_count = 0
        while frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
        
        cap.release()
        return frames
    
    def __len__(self):
        return len(self.frames) - self.frame_gap
    
    def __getitem__(self, index):
        present_frame = self.frames[index + self.frame_gap]
        past_frame = self.frames[index]
        
        present_tensor = self.transform(present_frame)
        past_tensor = self.transform(past_frame)
        
        return {
            'present': present_tensor,
            'past': past_tensor,
            'present_idx': index + self.frame_gap,
            'past_idx': index
        }


def test_video_pair_dataset(VIDEO_PATH):
    # Initialize the dataset
    dataset = VideoPairDataset(VIDEO_PATH, frame_gap=2, img_size=(192, 640), max_frames=30)

    # Sanity check: dataset length
    print("Dataset length:", len(dataset))
    
    # Check a few samples
    for idx in range(min(3, len(dataset))):
        sample = dataset[idx]
        print(f"Sample {idx}:")
        print("  present idx:", sample['present_idx'])
        print("  past idx:", sample['past_idx'])
        print("  present shape:", sample['present'].shape)
        print("  past shape:", sample['past'].shape)
        assert sample['present'].shape == torch.Size([3, 192, 640]), "Shape mismatch"
        assert sample['past'].shape == torch.Size([3, 192, 640]), "Shape mismatch"
        assert sample['present_idx'] == sample['past_idx'] + 2, "Frame gap mismatch"

    # Test DataLoader batching
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print("Batch present shape:", batch['present'].shape)
    print("Batch past shape:", batch['past'].shape)

if __name__ == "__main__":
    # Replace with actual path to your test video file
    load_dotenv()
    VIDEO_PATH = os.getenv('video_path')
    test_video_pair_dataset(VIDEO_PATH)
