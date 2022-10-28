import urllib.request
import sys
import time
import urllib.request
import zipfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def download_progress_hook(count, block_size, total_size):

    """
    A hook to report the progress of a download. Displays a progress bar.
    :param count: count of blocks downloaded so far
    :param block_size: size of each block
    :param total_size: total size of the file
    :return: None
    """
    global start_time

    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_and_extract(url, download_path, extract_path, hook=None):
    """
    Download the file at url, and extract it to extract_path.
    Args:
    - url (str): URL to download the file from
    - download_path (str): Path to download the file to
    - extract_path (str): Path to extract the file to
    Returns:
    - None
    """

    os.makedirs(download_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)
    file_name = os.path.join(download_path, os.path.basename(url))
    urllib.request.urlretrieve(url, file_name, reporthook=hook)
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def get_coco_dataloaders(coco_path, transform=None, batch_size=32):
    """
    Load the COCO dataset.
    Args:
    - coco_path (str): Path to the COCO dataset.
    - transform (torchvision.transforms): Transform to apply to the images.
    - batch_size (int): Batch size.
    Returns:
    - train_loader (torch.utils.data.DataLoader): Training data loader.
    - test_loader (torch.utils.data.DataLoader): Test data loader.
    """

    train_dataset = datasets.CocoKeypoints(
        root=os.path.join(coco_path, "train2017"),
        annFile=os.path.join(coco_path, "annotations/person_keypoints_train2017.json"),
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader