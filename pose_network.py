# from deepseek


import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO 1. add the batch norma layers, 
#add dropout layers, 
# use the efficient layers like depth wise seperable networks




class PoseNetwork(nn.Module):
    def __init__(self, num_pose_frames=2, rotation_mode='euler'):
        super(PoseNetwork, self).__init__()
        
        self.rotation_mode = rotation_mode

        # architecture 
        # conv1: 3 * num_pose_frames -> 32, kernel=7, stride=2, padding=3, followed by BN and ReLU
        # conv2: 32 -> 64, kernel=5, stride=2, padding=2, followed by BN and ReLU
        # conv3: 64 -> 128, kernel=3, stride=2, padding=1, followed by BN and ReLU
        # conv4: 128 -> 256, kernel=3, stride=2, padding=1, followed by BN and ReLU
        # conv5: 256 -> 256, kernel=3, stride=2, padding=1, followed by BN and ReLU
        # conv6: 256 -> 256, kernel=3, stride=2, padding=1, followed by BN and ReLU
        # conv7: 256 -> 256, kernel=3, stride=2, padding=1, followed by BN and ReLU
        # conv8: 256 -> 256, kernel=3, stride=2, padding=1, followed by BN and ReLU
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3 * num_pose_frames, 16, 7, 2, 3)
        self.conv2 = nn.Conv2d(16, 32, 5, 2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 2, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 2, 1)
        
        # Fully connected layers for pose prediction
        self.pose_fc = nn.Linear(256 * 4 * 8, 6 * (num_pose_frames - 1))
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, frames):
        """
        Args:
            frames: Tensor of shape [B, num_frames, C, H, W]
        
        Returns:
            poses: Tensor of shape [B, num_poses, 6] where num_poses = num_frames - 1
        """
        batch_size, num_frames, C, H, W = frames.shape
        
        # Concatenate consecutive frames along channel dimension
        x = frames.view(batch_size, num_frames * C, H, W)
        
        # Feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        
        # TODO Need to eliminate these global avergae pooling

        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (4, 8))
        x = x.view(batch_size, -1)
        
        # Pose prediction
        pose_params = self.pose_fc(x)
        poses = pose_params.view(batch_size, -1, 6)
        
        return poses

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example input: batch of 4 sequences, each with 2 frames of 3x128x256
    example_frames = torch.randn(4, 2, 3, 256, 256).to(device)
    
    # Initialize pose network
    pose_net = PoseNetwork(num_pose_frames=2).to(device)
    
    # Get predicted poses
    with torch.no_grad():
        predicted_poses = pose_net(example_frames)
    
    print(f"Output poses shape: {predicted_poses.shape}")  # Should be [4, 1, 6]