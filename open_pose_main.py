import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from helpers import download_and_extract, get_coco_dataloaders
from openpose_model import OpenPoseModel
import urllib.request

# URLs and Paths
coco_images_url = "http://images.cocodataset.org/zips/train2017.zip"
coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
pretrained_openpose_url = "http://images.cocodataset.org/zips/train2017.zip"
download_path = "downloads"
extract_path = "datasets"
coco_path = os.path.join(extract_path, "coco")
pretrained_model_path = os.path.join(download_path, "openpose.pth") 

# Download and set up datasets
download_and_extract(coco_images_url, download_path, coco_path)
download_and_extract(coco_annotations_url, download_path, coco_path)
if not os.path.exists(pretrained_model_path):
    urllib.request.urlretrieve(pretrained_openpose_url, pretrained_model_path)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
train_loader, test_loader = get_coco_dataloaders(coco_path, transform)

# Set up and train the model
model = OpenPoseModel()
model.load_state_dict(torch.load(pretrained_model_path))
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
