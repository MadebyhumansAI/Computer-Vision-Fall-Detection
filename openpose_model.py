import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
import torch

class OpenPoseModel(nn.Module):
    """
    A PyTorch implementation of the OpenPose model for human pose estimation.

    This model consists of a VGG-like backbone followed by two stages of convolutional layers.
    The first stage predicts the confidence maps for each keypoint, and the second stage refines
    the predictions by incorporating the features from the first stage.

    Attributes:
    - vgg_backbone (nn.Sequential): A sequential module representing the VGG-like backbone.
    - stage1 (nn.Sequential): A sequential module representing the first stage of convolutional layers.
    - stage2 (nn.Sequential): A sequential module representing the second stage of convolutional layers.
    - quant (QuantStub): A quantization stub for quantizing the input tensor.
    - dequant (DeQuantStub): A dequantization stub for dequantizing the output tensor.

    Methods:
    - forward(x): Performs a forward pass of the model on the input tensor x.
                  Returns the output tensor, which has shape (batch_size, num_keypoints, height, width).
    """
    def __init__(self):
        super(OpenPoseModel, self).__init__()

        # Basic VGG-like backbone
        self.vgg_backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 19, kernel_size=1, stride=1)  # Output 19 keypoints
        )
        
        self.stage2 = nn.Sequential(
            nn.Conv2d(64+19, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 19, kernel_size=1, stride=1)
        )
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        features = self.vgg_backbone(x)
        stage1_out = self.stage1(features)
        combined_features = torch.cat([features, stage1_out], 1)
        stage2_out = self.stage2(combined_features)
        
        x = self.dequant(stage2_out)
        return x
