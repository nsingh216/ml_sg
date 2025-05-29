import argparse
import os
import logging
import time
import torch
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AWS5SpeciesDataset(torch.utils.data.Dataset):
    """
    Dataset that filters for only the 5 species AWS used in their example.
    AWS used classes [17, 36, 47, 68, 73] from the CUB-200 dataset.
    """

    def __init__(self, root: str, annFile: str, transform=None, aws_classes_only=True):
        """
        Args:
            root (str): Path to the image directory.
            annFile (str): Path to COCO-style annotation JSON file.
            transform (callable, optional): Transform to apply to images.
            aws_classes_only (bool): If True, only use AWS's 5 classes [17, 36, 47, 68, 73]
        """
        self.root = root
        self.coco = COCO(annFile)
        self.transform = transform
        
        # AWS used these 5 specific classes for their 11-minute training
        self.aws_classes = [17, 36, 47, 68, 73]
        
        if aws_classes_only:
            # Filter to only images that contain one of the AWS classes
            all_ids = list(sorted(self.coco.imgs.keys()))
            self.ids = []
            
            for img_id in all_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                anns = self.coco.loadAnns(ann_ids)
                
                # Check if this image contains any of the AWS classes
                has_aws_class = False
                for ann in anns:
                    if ann['category_id'] in self.aws_classes:
                        has_aws_class = True
                        break
                
                if has_aws_class:
                    self.ids.append(img_id)
            
            print(f"Filtered dataset: {len(self.ids)} images with AWS's 5 species (vs {len(all_ids)} total)")
            
            # Create mapping from original class IDs to 0-4
            self.class_mapping = {cls_id: idx for idx, cls_id in enumerate(self.aws_classes)}
        else:
            self.ids = list(sorted(self.coco.imgs.keys()))
            self.class_mapping = None

    def __getitem__(self, index):
        """
        Retrieves the image and corresponding annotations at the given index.
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        boxes = []
        labels = []
        for ann in anns:
            category_id = ann['category_id']
            
            # If using AWS classes only, filter and remap
            if self.class_mapping is not None:
                if category_id in self.class_mapping:
                    xmin, ymin, width, height = ann['bbox']
                    boxes.append([xmin, ymin, xmin + width, ymin + height])
                    # Remap to 0-4 instead of original class IDs
                    labels.append(self.class_mapping[category_id])
            else:
                xmin, ymin, width, height = ann['bbox']
                boxes.append([xmin, ymin, xmin + width, ymin + height])
                labels.append(category_id)

        # Skip images with no valid boxes after filtering
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]  # Dummy box
            labels = [0]  # Dummy label

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
        return len(self.ids)


def get_transform(image_size=512):
    """Returns optimized image transformations."""
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
    Training pipeline comparing to AWS's 5-species example.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug(f"Using device: {device}")

    # Maximum CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 60)
    print("🐦 AWS 5-Species Comparison Training")
    print("=" * 60)
    print(f"AWS claimed: 100 epochs in 11 minutes for 5 species")
    print(f"Your settings: {args.epochs} epochs, batch size {args.batch_size}")
    print("=" * 60)

    # Create dataset with AWS's 5 species only
    dataset = AWS5SpeciesDataset(
        root=args.data_path,
        annFile=args.ann_file,
        transform=get_transform(args.image_size),
        aws_classes_only=True  # This is the key difference!
    )
    
    print(f"Dataset size: {len(dataset)} images (AWS's 5 species only)")
    
    # Optimized DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
        collate_fn=collate_fn
    )

    # Create model for 5 classes + background (6 total)
    num_classes = 6  # 5 bird species + 1 background
    model = create_model(num_classes, use_lightweight=args.use_lightweight)
    model.to(device)
    
    # Compile model if requested
    if hasattr(torch, 'compile') and args.use_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead', dynamic=True)
            print("✅ Model compiled successfully")
        except Exception as e:
            print(f"❌ Compilation failed: {e}")
            print("Continuing without compilation...")
    
    model.train()
    print(f"Model configured for {num_classes} classes (5 species + background)")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=0.0001)
    else:
        optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if args.use_amp else None
    accumulation_steps = args.accumulation_steps

    print(f"🚀 Starting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}, Accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * accumulation_steps}")
    print(f"Total batches per epoch: {len(data_loader)}")
    
    total_start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        running_loss = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            # Move data to device
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            if args.use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values()) / accumulation_steps
                
                scaler.scale(losses).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values()) / accumulation_steps
                losses.backward()
                
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

        epoch_time = time.time() - epoch_start_time
        print(f"✅ Epoch {epoch + 1} completed in {epoch_time/60:.1f} minutes")

    total_time = time.time() - total_start_time
    
    print("\n" + "=" * 60)
    print("🏁 TRAINING COMPLETE - AWS COMPARISON RESULTS")
    print("=" * 60)
    print(f"📊 Your Results:")
    print(f"   • Total time: {total_time/60:.1f} minutes")
    print(f"   • Time per epoch: {total_time/args.epochs/60:.1f} minutes")
    print(f"   • Projected 100 epochs: {total_time/args.epochs*100/60:.1f} minutes")
    print(f"")
    print(f"🎯 AWS Claimed (5 species):")
    print(f"   • 100 epochs: 11 minutes")
    print(f"   • Time per epoch: 0.11 minutes")
    print(f"")
    print(f"⚖️  Comparison:")
    if args.epochs >= 1:
        your_time_per_epoch = total_time / args.epochs / 60
        aws_time_per_epoch = 0.11
        ratio = your_time_per_epoch / aws_time_per_epoch
        print(f"   • Your PyTorch vs AWS SageMaker: {ratio:.1f}x slower")
        print(f"   • Dataset size: {len(dataset)} images")
        print(f"   • Your approach: Faster R-CNN (more accurate)")
        print(f"   • AWS approach: SSD (faster but less accurate)")
    print("=" * 60)

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "aws_comparison_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AWS 5-Species Comparison Training')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='CUB_200_2011/images', help='Path to image folder')
    parser.add_argument('--ann-file', type=str, default='annotations.json', help='Path to COCO annotation JSON file')
    parser.add_argument('--model-dir', type=str, default='./models', help='Directory to save model')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--accumulation-steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=0.005, help='Learning rate for optimizer')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw'], help='Optimizer type')
    
    # Optimization arguments
    parser.add_argument('--use-amp', action='store_true', help='Use automatic mixed precision')
    parser.add_argument('--use-compile', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--use-lightweight', action='store_true', help='Use MobileNet backbone instead of ResNet50')
    parser.add_argument('--image-size', type=int, default=512, help='Input image size (default: 512)')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=25, help='Log every N batches')
    
    args = parser.parse_args()
    main(args)


# USAGE EXAMPLES:

# Test 1 epoch to see the speed:
# python aws_comparison.py --data-path CUB_200_2011/images --ann-file annotations.json --epochs 1 --use-amp

# Compare directly to AWS's claim (100 epochs):
# python aws_comparison.py --data-path CUB_200_2011/images --ann-file annotations.json --epochs 100 --use-amp --use-compile

# Ultra-fast version with lightweight model:
# python aws_comparison.py --data-path CUB_200_2011/images --ann-file annotations.json --epochs 100 --use-amp --use-compile --use-lightweight --batch-size 32
