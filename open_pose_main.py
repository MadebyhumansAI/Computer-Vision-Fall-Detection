import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from helpers import download_and_extract, get_coco_dataloaders
from openpose_model import OpenPoseModel
import urllib.request

def download_datasets(images_url, annotations_url, download_path, extract_path):
    """
    Download and extract the COCO datasets.

    Args:
    - images_url (str): URL of the COCO images dataset.
    - annotations_url (str): URL of the COCO annotations dataset.
    - download_path (str): Path to the directory where the datasets will be downloaded.
    - extract_path (str): Path to the directory where the datasets will be extracted.
    """
    download_and_extract(images_url, download_path, extract_path)
    download_and_extract(annotations_url, download_path, extract_path)

def download_pretrained_model(model_url, model_path):
    """
    Download a pretrained OpenPose model.

    Args:
    - model_url (str): URL of the pretrained OpenPose model.
    - model_path (str): Path to the file where the model will be saved.
    """
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(model_url, model_path)

def setup_dataloaders(data_path, transform):
    """
    Set up the COCO dataloaders.

    Args:
    - data_path (str): Path to the directory where the COCO datasets are stored.
    - transform (torchvision.transforms.Compose): A composition of PyTorch transforms to apply to the input images.

    Returns:
    - train_loader (torch.utils.data.DataLoader): A PyTorch dataloader for the COCO training dataset.
    - test_loader (torch.utils.data.DataLoader): A PyTorch dataloader for the COCO test dataset.
    """
    train_loader, test_loader = get_coco_dataloaders(data_path, transform)
    return train_loader, test_loader

def load_pretrained_model(model_path):
    """
    Load a pretrained OpenPose model.

    Args:
    - model_path (str): Path to the file where the model is stored.

    Returns:
    - model (openpose_model.OpenPoseModel): An instance of the OpenPoseModel class with the pretrained weights loaded.
    """
    model = OpenPoseModel()
    model.load_state_dict(torch.load(model_path))
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    return model

if __name__ == '__main__':
    # URLs and Paths
    coco_images_url = "http://images.cocodataset.org/zips/train2017.zip"
    coco_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    pretrained_openpose_url = "http://images.cocodataset.org/zips/train2017.zip"
    download_path = "downloads"
    extract_path = "datasets"
    coco_path = os.path.join(extract_path, "coco")
    pretrained_model_path = os.path.join(download_path, "openpose.pth") 

    # Download and set up datasets
    download_datasets(coco_images_url, coco_annotations_url, download_path, extract_path)

    # Download pretrained model
    download_pretrained_model(pretrained_openpose_url, pretrained_model_path)

    # Set up dataloaders
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    train_loader, test_loader = setup_dataloaders(coco_path, transform)

    # Load pretrained model
    model = load_pretrained_model(pretrained_model_path)
