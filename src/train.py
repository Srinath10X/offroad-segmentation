"""
Segmentation Training Script - TerraSeg v3
Trains a segmentation head on top of DINOv2 backbone
Optimized with AdamW (lr=1e-3) and Cosine Annealing LR Scheduler
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import torchvision
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style)
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )

        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())
    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score = (2. * intersection + smooth) / (pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())
    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    iou_scores, dice_scores, pixel_accuracies = [], [], []
    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            output = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = model(output.to(device))
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            labels = labels.squeeze(dim=1).long()
            iou_scores.append(compute_iou(outputs, labels, num_classes=num_classes))
            dice_scores.append(compute_dice(outputs, labels, num_classes=num_classes))
            pixel_accuracies.append(compute_pixel_accuracy(outputs, labels))
    model.train()
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies)


# ============================================================================
# Plotting & Stats Functions
# ============================================================================

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics = [('Loss', 'train_loss', 'val_loss'), ('IoU', 'train_iou', 'val_iou'), 
               ('Dice Score', 'train_dice', 'val_dice'), ('Pixel Accuracy', 'train_pixel_acc', 'val_pixel_acc')]
    
    plt.figure(figsize=(15, 12))
    for i, (name, train_key, val_key) in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        plt.plot(history[train_key], label='train')
        plt.plot(history[val_key], label='val')
        plt.title(f'{name} vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()


def save_history_to_file(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("TERRASEG V2 TRAINING RESULTS\n" + "="*50 + "\n")
        f.write(f"Best Val IoU: {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou']) + 1})\n")
        f.write("-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        for i in range(len(history['train_loss'])):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1, history['train_loss'][i], history['val_loss'][i], history['train_iou'][i], 
                history['val_iou'][i], history['train_pixel_acc'][i], history['val_pixel_acc'][i]))


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 8
    n_epochs = 20
    lr = 1e-3
    w, h = 476, 266 # Optimized for DINOv2 14x14 patches

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    # Dataset Paths
    data_dir = os.path.join(script_dir, '..', 'data', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir = os.path.join(script_dir, '..', 'data', 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset = MaskDataset(data_dir=data_dir, transform=transform, mask_transform=mask_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = MaskDataset(data_dir=val_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    # Load Backbone
    print("Loading DINOv2 backbone...")
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vits14")
    backbone_model.eval().to(device)

    # Get dimensions
    imgs, _ = next(iter(train_loader))
    with torch.no_grad():
        output = backbone_model.forward_features(imgs.to(device))["x_norm_patchtokens"]
    n_embedding = output.shape[2]

    # Initialize Head
    classifier = SegmentationHeadConvNeXt(in_channels=n_embedding, out_channels=n_classes, 
                                          tokenW=w // 14, tokenH=h // 14).to(device)

    loss_fct = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    history = {k: [] for k in ['train_loss', 'val_loss', 'train_iou', 'val_iou',
                               'train_dice', 'val_dice', 'train_pixel_acc', 'val_pixel_acc']}

    print("\nStarting Training (TerraSeg v2)...")
    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        classifier.train()
        train_losses = []
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feat = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(feat)
            preds = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            loss = loss_fct(preds, labels.squeeze(1).long())
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        classifier.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                preds = F.interpolate(classifier(feat), size=imgs.shape[2:], mode="bilinear", align_corners=False)
                val_losses.append(loss_fct(preds, labels.squeeze(1).long()).item())

        # Metrics
        t_iou, t_dice, t_acc = evaluate_metrics(classifier, backbone_model, train_loader, device)
        v_iou, v_dice, v_acc = evaluate_metrics(classifier, backbone_model, val_loader, device)

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_iou'].append(t_iou); history['val_iou'].append(v_iou)
        history['train_dice'].append(t_dice); history['val_dice'].append(v_dice)
        history['train_pixel_acc'].append(t_acc); history['val_pixel_acc'].append(v_acc)

        scheduler.step()
        epoch_pbar.set_postfix(val_iou=f"{v_iou:.3f}", val_acc=f"{v_acc:.3f}")

    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)
    torch.save(classifier.state_dict(), os.path.join(script_dir, "segmentation_head.pth"))
    print(f"\nTraining Complete. Model saved to src/segmentation_head.pth")

if __name__ == "__main__":
    main()
