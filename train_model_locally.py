import argparse
import os
import logging
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

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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


class OptimizedCocoDataset(torch.utils.data.Dataset):
    """
    Optimized PyTorch Dataset for loading COCO-style annotated images.
    """

    def __init__(self, root: str, annFile: str, transform=None, cache_images=False):
        """
        Args:
            root (str): Path to the image directory.
            annFile (str): Path to COCO-style annotation JSON file.
            transform (callable, optional): Transform to apply to images.
            cache_images (bool): Whether to cache images in memory.
        """
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.cache_images = cache_images
        self._image_cache = {}
        
        # Pre-compute image paths and annotations
        self._precompute_data()

    def _precompute_data(self):
        """Pre-compute image paths and annotations for faster access."""
        self.image_data = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            path = self.coco.loadImgs(img_id)[0]['file_name']
            
            boxes = []
            labels = []
            for ann in anns:
                xmin, ymin, width, height = ann['bbox']
                boxes.append([xmin, ymin, xmin + width, ymin + height])
                labels.append(ann['category_id'])
            
            self.image_data.append({
                'path': os.path.join(self.root, path),
                'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                'labels': torch.as_tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([img_id])
            })

    def __getitem__(self, index):
        """
        Retrieves the image and corresponding annotations at the given index.
        """
        data = self.image_data[index]
        
        # Check cache first
        if self.cache_images and index in self._image_cache:
            img = self._image_cache[index]
        else:
            img = Image.open(data['path']).convert("RGB")
            if self.cache_images:
                self._image_cache[index] = img

        target = {
            "boxes": data['boxes'],
            "labels": data['labels'],
            "image_id": data['image_id'],
        }

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.ids)


def get_transform(image_size=512):  # Reduced from 800 for speed
    """
    Returns optimized image transformations.
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])


def collate_fn(batch):
    """Custom collate function for handling batches with variable-sized targets."""
    return tuple(zip(*batch))


def create_model(num_classes, use_lightweight=False):
    """Create and configure the model."""
    if use_lightweight:
        # Use MobileNet backbone for faster training
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
        model = fasterrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def main(args):
    """
    Ultra-optimized training pipeline for object detection.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")

    # Maximum CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)

    # Compile model if using PyTorch 2.0+
    if hasattr(torch, 'compile') and args.use_compile:
        print("Using torch.compile for model optimization")

    dataset = OptimizedCocoDataset(
        root=args.data_path,
        annFile=args.ann_file,
        transform=get_transform(args.image_size),
        cache_images=args.cache_images
    )
    logger.debug(f"Loaded {len(dataset)} images from {args.data_path}")

    # Optimized DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(16, os.cpu_count()),  # Use more workers
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=8,  # Increased prefetch
        drop_last=True,  # Consistent batch sizes for optimization
        collate_fn=collate_fn
    )

    # Create model
    model = create_model(201, use_lightweight=args.use_lightweight)
    model.to(device)
    
    # Compile model for PyTorch 2.0+ speed boost (with safer options for object detection)
    if hasattr(torch, 'compile') and args.use_compile:
        try:
            # Use a more conservative compilation mode for object detection
            model = torch.compile(model, mode='reduce-overhead', dynamic=True)
            print("Successfully compiled model with dynamic=True")
        except Exception as e:
            print(f"Compilation failed: {e}")
            print("Continuing without compilation...")
    
    model.train()
    logger.debug("Model loaded and optimized.")

    # Optimized optimizer with larger learning rate for bigger batches
    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.0001)
    else:
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1) if args.use_scheduler else None
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None

    # Gradient accumulation setup
    accumulation_steps = args.accumulation_steps

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * accumulation_steps}")
    
    total_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        
        optimizer.zero_grad()  # Zero gradients at start of epoch
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            # Move data to device with non_blocking transfer
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            if args.use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values()) / accumulation_steps
                
                scaler.scale(losses).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values()) / accumulation_steps
                losses.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            running_loss += losses.item() * accumulation_steps

            # Logging
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                elapsed = time.time() - epoch_start_time
                batches_per_sec = (batch_idx + 1) / elapsed
                eta_seconds = (len(data_loader) - batch_idx - 1) / batches_per_sec
                
                print(f"Epoch {epoch + 1}/{args.epochs}, Batch {batch_idx + 1}/{len(data_loader)}, "
                      f"Loss: {avg_loss:.4f}, Speed: {batches_per_sec:.1f} batch/s, "
                      f"ETA: {eta_seconds/60:.1f}min")
                running_loss = 0.0

        # Step scheduler
        if scheduler:
            scheduler.step()

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time/60:.1f} minutes")

        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
            }
            if scaler:
                checkpoint['scaler_state_dict'] = scaler.state_dict()
            
            # Save both numbered and latest checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            latest_path = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
            
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, latest_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        # Save checkpoint every few epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.model_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            os.makedirs(args.model_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    total_time = time.time() - total_start_time
    print(f"\nTotal training completed in {total_time/3600:.1f} hours")

    # Save final model
    os.makedirs(args.model_dir, exist_ok=True)
    final_model_path = os.path.join(args.model_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.debug(f"Final model saved to {final_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ultra-optimized Faster R-CNN training')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='CUB_200_2011/images', help='Path to image folder')
    parser.add_argument('--ann-file', type=str, default='annotations.json', help='Path to COCO annotation JSON file')
    parser.add_argument('--model-dir', type=str, default='./models', help='Directory to save model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--accumulation-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for optimizer')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'], help='Optimizer type')
    
    # Optimization arguments
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--use-compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--use-lightweight', action='store_true', help='Use MobileNet backbone instead of ResNet50')
    parser.add_argument('--use-scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--cache-images', action='store_true', help='Cache images in memory (requires lots of RAM)')
    parser.add_argument('--image-size', type=int, default=512, help='Input image size (default: 512)')
    
    # Logging and saving
    parser.add_argument('--log-interval', type=int, default=100, help='Log every N batches')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)


# SPEED OPTIMIZATION COMMANDS:

# Ultra-fast training (may sacrifice some accuracy):
# python train.py --data-path CUB_200_2011/images --ann-file annotations.json --batch-size 24 --epochs 100 --use-amp --use-compile --use-lightweight --image-size 416 --accumulation-steps 2 --optimizer adamw --learning-rate 0.001 --use-scheduler

# Balanced speed/accuracy:
# python train.py --data-path CUB_200_2011/images --ann-file annotations.json --batch-size 16 --epochs 100 --use-amp --use-compile --image-size 512 --accumulation-steps 2 --optimizer sgd --learning-rate 0.01

# Maximum speed (if you have lots of RAM):
# python train.py --data-path CUB_200_2011/images --ann-file annotations.json --batch-size 32 --epochs 100 --use-amp --use-compile --use-lightweight --cache-images --image-size 384 --accumulation-steps 1 --optimizer adamw --learning-rate 0.001
