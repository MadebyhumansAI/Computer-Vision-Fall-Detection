import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
import torchvision.transforms as transforms
from pathlib import Path
import torch.optim as optim
from yolov5.utils.loss import ComputeLoss  # Ultralytics specific YOLO loss
from torch.quantization import get_default_qconfig, prepare_qat
# import yolo5_hyper_parameters as hyp
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from helpers import get_yolo_hyperparameters

def load_yolo_model(model_name='yolov5s', pretrained=True):
    """Load a YOLOv5 model using Ultralytics implementation.

    Args:
    - model_name (str): Name of the YOLO model variant ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    - pretrained (bool): If True, loads the pretrained weights.

    Returns:
    - model (torch.nn.Module): YOLOv5 model.
    """

    # Path to YOLOv5 pre-trained weights
    model_urls = {
        'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt',
        'yolov5m': 'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt',
        'yolov5l': 'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt',
        'yolov5x': 'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt'
    }

    # Construct path to cached model weights
    cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_file = cache_dir / f"{model_name}.pt"


    if pretrained:
        if not model_file.exists():
            # Download the model weights if not already available
            torch.hub.download_url_to_file(model_urls[model_name], model_file)
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=False)

    return model


def load_quantized_yolo_model(model_name='yolov5s', pretrained=True):
    """
    Load a YOLOv5 model with quantization stubs added.

    Args:
    - model_name (str): Name of the YOLO model variant ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']
    - pretrained (bool): If True, loads the pretrained weights.

    Returns:
    - model (torch.nn.Module): YOLOv5 model with quantization stubs.
    """
    model = load_yolo_model(model_name, pretrained)
    
    # Add quantization and dequantization stubs
    model.quant = QuantStub() # emulate quantizing int8
    model.dequant = DeQuantStub() # emulate dequantizing float32

    return model

def prepare_data():
    """Load and prepare the VOC dataset."""

    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # YOLOv5 uses 640 as a default size
        transforms.ToTensor(),
    ])

    train_dataset = VOCDetection(root="./data", year="2012", image_set="train", download=True, transform=transform)
    test_dataset = VOCDetection(root="./data", year="2012", image_set="val", download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, test_loader

def prepare_for_qat(model)->torch.nn.Module:
    """
    Prepare a YOLOv5 model for quantization aware training.
    
    Args:
    - model (torch.nn.Module): The model to prepare for quantization aware training.

    Returns:
    - model (torch.nn.Module): The model prepared for quantization aware training.

    """
    
    qconfig = get_default_qconfig('fbgemm') #get config for 8-bit quantization kernel
    model.qconfig = qconfig #set the qconfig for the model
    torch.backends.quantized.engine = 'fbgemm' #set the backend engine for quantization to fb-gemm, you can choose from qnnpack or fb-gemm
    model = prepare_qat(model) # pass to torch library for preparing the model for quantization aware training
    return model

def compute_yolo_loss(outputs, targets, anchors, num_classes, image_size=640):
    """
    Compute the YOLO loss given the model's outputs and ground-truth targets.

    Args:
    - outputs (tensor): The model's predictions.
    - targets (tensor): Ground truth labels.
    - anchors (list): Anchor box sizes.
    - num_classes (int): Number of classes.
    - image_size (int): Size of input images.

    Returns:
    - loss (tensor): Total YOLO loss.
    """

    # Split outputs into individual components
    pred_boxes, objectness, class_probs = torch.split(outputs, [4, 1, num_classes], dim=-1)

    # Ground truth components
    target_boxes, obj_mask, class_mask = targets

    # 1. Localization Loss
    pred_coords = pred_boxes[..., :2] * obj_mask
    pred_wh = pred_boxes[..., 2:4] * obj_mask
    target_coords = target_boxes[..., :2]
    target_wh = target_boxes[..., 2:4]
    coord_loss = F.mse_loss(pred_coords, target_coords) + F.mse_loss(pred_wh, target_wh)

    # 2. Objectness Loss
    obj_loss = F.binary_cross_entropy_with_logits(objectness, obj_mask)

    # 3. Classification Loss
    class_loss = F.cross_entropy(class_probs[obj_mask], class_mask[obj_mask])

    # Combine the losses
    loss = coord_loss + obj_loss + class_loss

    return loss

def train_qat(model, train_loader, num_epochs=5) -> torch.nn.Module:

    """
    Train a quantized YOLOv5 model.

    Args:
    - model (torch.nn.Module): The model to train.
    - train_loader (torch.utils.data.DataLoader): Training data loader.
    - num_epochs (int): Number of epochs to train for.

    Returns:
    - model (torch.nn.Module): Trained model.
    
    """

    # Define loss and optimizer
    # Parameters:
    # model.parameters() - Retrieves the parameters (weights & biases) of the given model.
    # lr=hyp['lr0']       - The learning rate which determines the step size at each iteration 
    #                       while moving toward a minimum of the loss function. It's one of the hyperparameters 
    #                      we can tune, and in this case, it's fetched from a dictionary named 'hyp'.

    hyp = get_yolo_hyperparameters()
    
    optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'])

    criterion = ComputeLoss(model, hyp)  # Ultralytics specific YOLO loss

    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_items = criterion(outputs, targets)  # Using Ultralytics loss function
            loss.backward()
            optimizer.step()

    # Convert the model to use quantized operations
    model = torch.quantization.convert(model.eval(), inplace=False)


    return model


if __name__ == "__main__":
    model = load_quantized_yolo_model('yolov5s', pretrained=True)  # take small model for faster training and inference
    model = prepare_for_qat(model)

    train_loader, _ = prepare_data()
    
    trained_model = train_qat(model, train_loader, num_epochs=5)
    
    # Save the trained model's checkpoint
    torch.save(trained_model.state_dict(), 'trained_model_checkpoint.pth')


