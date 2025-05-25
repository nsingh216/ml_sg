import argparse
import os
import shutil
import tarfile
import time
import torch
import urllib.request

from datetime import datetime
from torch.utils.data import DataLoader
from PIL import Image
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Use the default pretrained weights for Faster R-CNN with ResNet50 backbone
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT


def download(url: str) -> None:
    """
    Downloads a file from a given URL if it doesn't already exist.

    Args:
        url (str): The URL to download from.
    """
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists, skipping download.")


def extract_cub_dataset(tgz_path: str = "./CUB_200_2011.tgz", extract_dir: str = ".") -> None:
    """
    Extracts the CUB_200_2011 dataset from a .tgz file and deletes the archive afterward.

    Args:
        tgz_path (str): Path to the .tgz file.
        extract_dir (str): Directory where the archive should be extracted.
    """
    cub_dir = os.path.join(extract_dir, "CUB_200_2011")

    if os.path.exists(cub_dir):
        print(f"Removing existing directory: {cub_dir}")
        shutil.rmtree(cub_dir)

    print(f"Extracting {tgz_path} to {extract_dir}...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    print(f"Removing archive: {tgz_path}")
    os.remove(tgz_path)


class CustomCocoDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset for loading COCO-style annotated images.
    """

    def __init__(self, root: str, annFile: str, transform=None):
        """
        Args:
            root (str): Path to the image directory.
            annFile (str): Path to COCO-style annotation JSON file.
            transform (callable, optional): Transform to apply to images.
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        """
        Retrieves the image and corresponding annotations at the given index.

        Args:
            index (int): Index of the item to fetch.

        Returns:
            Tuple[Tensor, Dict]: Transformed image and target dictionary.
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            xmin, ymin, width, height = ann['bbox']
            boxes.append([xmin, ymin, xmin + width, ymin + height])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        """
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.ids)


def get_transform():
    """
    Returns basic image transformations for resizing and tensor conversion.

    Returns:
        torchvision.transforms.Compose: Image transformations.
    """
    return T.Compose([
        T.Resize((300, 300)),
        T.ToTensor()
    ])


def collate_fn(batch):
    """
    Custom collate function for handling batches with variable-sized targets.

    Args:
        batch (list): List of (image, target) tuples.

    Returns:
        Tuple: Batch of images and targets.
    """
    return tuple(zip(*batch))


def main(args):
    """
    Main training pipeline for object detection using Faster R-CNN.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # download("https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz")
    # extract_cub_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dataset = CustomCocoDataset(
        root=args.data_path,
        annFile=args.ann_file,
        transform=get_transform()
    )
    print(f"Loaded {len(dataset)} images from {args.data_path}")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    print("DataLoader initialized.")

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

    # Replace head with custom number of classes (200 birds + 1 background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 201)

    model.to(device)
    model.train()
    print("Model loaded and set to training mode.")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)

    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        x = 1
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if x % 100 == 0:
                print(f"Epoch {epoch + 1}, x {x}, Loss: {losses.item():.4f}")
                print(f"completed {x} of {len(data_loader)} at {datetime.now()}")

            x += 1

        print(f"Epoch {epoch + 1}, Loss: {losses.item():.4f}, completed in {time.time() - start_time:.2f} seconds")   

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Faster R-CNN on a custom COCO-style dataset')
    parser.add_argument('--data-path', type=str, default='CUB_200_2011/images', help='Path to image folder')
    parser.add_argument('--ann-file', type=str, default='annotations.json', help='Path to COCO annotation JSON file')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model', help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.005, help='Learning rate for optimizer')
    args = parser.parse_args()

    main(args)


# pip install pycocotools
# python train.py   --data-path CUB_200_2011/images   --ann-file annotations.json   --batch-size 2   --epochs 1


# Epoch 1/1
# Epoch 1, x 100, Loss: 0.1240
# completed 100 of 737 at 2025-05-22 00:56:25.159439
# Epoch 1, x 200, Loss: 0.1443
# completed 200 of 737 at 2025-05-22 00:57:51.963413
# Epoch 1, x 300, Loss: 0.1674
# completed 300 of 737 at 2025-05-22 00:59:18.996707
# Epoch 1, x 400, Loss: 0.1413
# completed 400 of 737 at 2025-05-22 01:00:45.812100
# Epoch 1, x 500, Loss: 0.1140
# completed 500 of 737 at 2025-05-22 01:02:12.941215
# Epoch 1, x 600, Loss: 0.0966
# completed 600 of 737 at 2025-05-22 01:03:39.937438
# Epoch 1, x 700, Loss: 0.1909
# completed 700 of 737 at 2025-05-22 01:05:06.970245
# Epoch 1, Loss: 0.1401, completed in 641.13 seconds
