"""
Segmentation Training Script - TerraSeg v3 (Fixed)
Key fixes over v2:
  1. Added Flowers (600 -> 6) to value_map — was silently collapsing to background
  2. Renumbered all class IDs correctly (10 classes total incl. Flowers)
  3. Weighted CrossEntropy to handle class imbalance
  4. Added data augmentation (RandomHFlip, ColorJitter)
  5. Fixed compute_iou: background (class 0) is now included in mean
  6. Added per-class IoU logging so you can track what's hurting the mean
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import random
from tqdm import tqdm

# ============================================================================
# Mask Conversion  —  NOW INCLUDES FLOWERS (600)
# ============================================================================

value_map = {
    0:     0,   # background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers   ← was MISSING; pixels were bleeding into background (0)
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}
n_classes = len(value_map)   # 11

CLASS_NAMES = [
    "Background", "Trees", "Lush Bushes", "Dry Grass",
    "Dry Bushes", "Ground Clutter", "Flowers", "Logs",
    "Rocks", "Landscape", "Sky"
]

# Rough inverse-frequency weights (higher = rarer class gets more weight).
# Sky / Landscape dominate → lower weight; rare classes → higher weight.
CLASS_WEIGHTS = torch.tensor([
    0.5,   # Background   — almost never correct label
    3.0,   # Trees
    5.0,   # Lush Bushes  — very rare
    2.5,   # Dry Grass
    3.0,   # Dry Bushes
    4.0,   # Ground Clutter
    6.0,   # Flowers      — rarest
    4.0,   # Logs
    3.5,   # Rocks
    1.0,   # Landscape
    0.8,   # Sky
], dtype=torch.float32)


def convert_mask(mask):
    """Convert raw mask pixel values to consecutive class IDs."""
    arr = np.array(mask, dtype=np.int32)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def save_image(img, filename):
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1)
    img  = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Dataset  —  with optional joint augmentation
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, img_size=(476, 266), augment=False):
        self.image_dir  = os.path.join(data_dir, 'Color_Images')
        self.masks_dir  = os.path.join(data_dir, 'Segmentation')
        self.data_ids   = os.listdir(self.image_dir)
        self.img_size   = img_size   # (W, H)
        self.augment    = augment

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id  = self.data_ids[idx]
        image    = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask     = Image.open(os.path.join(self.masks_dir, data_id))
        mask     = convert_mask(mask)

        W, H = self.img_size
        image = image.resize((W, H), Image.BILINEAR)
        mask  = mask.resize((W, H),  Image.NEAREST)

        # ── Joint augmentation (train only) ──────────────────────────────
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
            if random.random() > 0.5:
                image = TF.adjust_saturation(image, random.uniform(0.7, 1.3))

        image = self.img_transform(image)
        mask  = torch.from_numpy(np.array(mask)).long()   # (H, W), values 0-10
        return image, mask


# ============================================================================
# Model
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, padding=3, groups=256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=n_classes):
    """Returns (mean_iou, per_class_iou_list)."""
    pred_cls = torch.argmax(pred, dim=1).view(-1)
    target   = target.view(-1)

    per_class = []
    for c in range(num_classes):
        p = pred_cls == c
        t = target   == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        per_class.append(float('nan') if union == 0 else (inter / union).item())

    mean = float(np.nanmean(per_class))
    return mean, per_class


def compute_pixel_accuracy(pred, target):
    pred_cls = torch.argmax(pred, dim=1)
    return (pred_cls == target).float().mean().item()


@torch.no_grad()
def evaluate(model, backbone, loader, device):
    model.eval()
    ious, accs = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        feat   = backbone.forward_features(imgs)["x_norm_patchtokens"]
        logits = model(feat)
        preds  = F.interpolate(logits, size=imgs.shape[2:],
                               mode="bilinear", align_corners=False)
        miou, _ = compute_iou(preds, labels)
        ious.append(miou)
        accs.append(compute_pixel_accuracy(preds, labels))
    model.train()
    return float(np.nanmean(ious)), float(np.mean(accs))


@torch.no_grad()
def evaluate_per_class(model, backbone, loader, device):
    """Full per-class IoU averaged over the whole dataset."""
    model.eval()
    all_per_class = [[] for _ in range(n_classes)]
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        feat   = backbone.forward_features(imgs)["x_norm_patchtokens"]
        logits = model(feat)
        preds  = F.interpolate(logits, size=imgs.shape[2:],
                               mode="bilinear", align_corners=False)
        _, per_class = compute_iou(preds, labels)
        for c, v in enumerate(per_class):
            if not np.isnan(v):
                all_per_class[c].append(v)
    model.train()
    return [float(np.mean(v)) if v else float('nan') for v in all_per_class]


# ============================================================================
# Plotting
# ============================================================================

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    pairs = [
        ('Loss',           'train_loss',      'val_loss'),
        ('mIoU',           'train_iou',       'val_iou'),
        ('Pixel Accuracy', 'train_pixel_acc', 'val_pixel_acc'),
    ]
    for ax, (name, tk, vk) in zip(axes.flat, pairs):
        ax.plot(history[tk], label='train')
        ax.plot(history[vk], label='val')
        ax.set_title(f'{name} vs Epoch')
        ax.set_xlabel('Epoch'); ax.set_ylabel(name)
        ax.legend(); ax.grid(True)
    axes.flat[-1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()


def save_per_class_bar(per_class_iou, output_dir, title="Per-Class IoU"):
    os.makedirs(output_dir, exist_ok=True)
    colors = ['#888888','#2d5a27','#5aba46','#c8a84b','#7a5c2e',
              '#b0b030','#e060a0','#7a4010','#888090','#c47830','#60b8e0']
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(CLASS_NAMES, per_class_iou, color=colors)
    mean = float(np.nanmean(per_class_iou))
    ax.axhline(mean, color='red', linestyle='--', label=f'Mean {mean:.4f}')
    ax.set_title(title)
    ax.set_ylabel('IoU'); ax.set_ylim(0, 1)
    ax.legend(); plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'))
    plt.close()


def save_history_to_file(history, best_val_iou, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(path, 'w') as f:
        f.write("TERRASEG V3 TRAINING RESULTS\n" + "="*60 + "\n")
        f.write(f"Best Val mIoU : {best_val_iou:.4f}\n")
        f.write(f"Target mIoU   : 0.2476\n")
        f.write("-"*90 + "\n")
        hdr = ['Epoch','TrainLoss','ValLoss','TrainIoU','ValIoU','TrainAcc','ValAcc']
        f.write("{:<7} {:<11} {:<11} {:<11} {:<11} {:<11} {:<11}\n".format(*hdr))
        for i in range(len(history['train_loss'])):
            f.write("{:<7} {:<11.4f} {:<11.4f} {:<11.4f} {:<11.4f} {:<11.4f} {:<11.4f}\n".format(
                i+1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i],  history['val_iou'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i],
            ))


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Hyper-parameters ────────────────────────────────────────────────────
    batch_size = 8
    n_epochs   = 25          # a few more epochs
    lr         = 1e-3
    W, H       = 476, 266   # must be divisible by 14 for DINOv2 patches

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────────────────────
    data_dir = os.path.join(script_dir, '..', 'data',
                            'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, '..', 'data',
                            'Offroad_Segmentation_Training_Dataset', 'val')

    trainset    = MaskDataset(data_dir, img_size=(W, H), augment=True)
    valset      = MaskDataset(val_dir,  img_size=(W, H), augment=False)
    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(valset,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── Backbone ─────────────────────────────────────────────────────────────
    print("Loading DINOv2 backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Probe embedding dim
    with torch.no_grad():
        sample_imgs, _ = next(iter(train_loader))
        feat = backbone.forward_features(sample_imgs[:1].to(device))["x_norm_patchtokens"]
    n_emb = feat.shape[2]
    print(f"DINOv2 embedding dim: {n_emb}")

    # ── Segmentation head ────────────────────────────────────────────────────
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_emb,
        out_channels=n_classes,
        tokenW=W // 14,
        tokenH=H // 14,
    ).to(device)

    loss_fct  = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS.to(device),
        ignore_index=255,
    )
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    history = {k: [] for k in [
        'train_loss', 'val_loss',
        'train_iou',  'val_iou',
        'train_pixel_acc', 'val_pixel_acc',
    ]}
    best_val_iou   = 0.0
    best_ckpt_path = os.path.join(script_dir, "segmentation_head_best.pth")

    print(f"\nStarting Training — TerraSeg v3  (target mIoU ≥ 0.2476)\n")

    for epoch in range(n_epochs):
        classifier.train()
        train_losses, train_ious, train_accs = [], [], []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                feat = backbone.forward_features(imgs)["x_norm_patchtokens"]

            logits = classifier(feat)
            preds  = F.interpolate(logits, size=imgs.shape[2:],
                                   mode="bilinear", align_corners=False)
            loss = loss_fct(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                miou, _ = compute_iou(preds, labels)
                acc      = compute_pixel_accuracy(preds, labels)
            train_losses.append(loss.item())
            train_ious.append(miou)
            train_accs.append(acc)
            pbar.set_postfix(loss=f"{loss.item():.3f}", iou=f"{miou:.3f}")

        # ── Validation ───────────────────────────────────────────────────────
        val_losses = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feat  = backbone.forward_features(imgs)["x_norm_patchtokens"]
                preds = F.interpolate(classifier(feat), size=imgs.shape[2:],
                                      mode="bilinear", align_corners=False)
                val_losses.append(loss_fct(preds, labels).item())

        v_iou, v_acc = evaluate(classifier, backbone, val_loader, device)
        t_iou = float(np.mean(train_ious))
        t_acc = float(np.mean(train_accs))

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_iou'].append(t_iou)
        history['val_iou'].append(v_iou)
        history['train_pixel_acc'].append(t_acc)
        history['val_pixel_acc'].append(v_acc)

        scheduler.step()

        flag = " ← best" if v_iou > best_val_iou else ""
        print(f"Epoch {epoch+1:>3} | "
              f"Loss {np.mean(train_losses):.4f}/{np.mean(val_losses):.4f} | "
              f"mIoU {t_iou:.4f}/{v_iou:.4f} | "
              f"Acc {t_acc:.4f}/{v_acc:.4f}{flag}")

        if v_iou > best_val_iou:
            best_val_iou = v_iou
            torch.save(classifier.state_dict(), best_ckpt_path)

    # ── Final per-class IoU ──────────────────────────────────────────────────
    print("\nComputing final per-class IoU on validation set...")
    # Load best checkpoint for final eval
    classifier.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    per_class = evaluate_per_class(classifier, backbone, val_loader, device)
    print("\nPer-class IoU (best checkpoint):")
    for name, iou in zip(CLASS_NAMES, per_class):
        bar = "█" * int(iou * 40) if not np.isnan(iou) else ""
        print(f"  {name:<18} {iou:.4f}  {bar}")
    print(f"\n  Mean IoU: {float(np.nanmean(per_class)):.4f}  (target: 0.2476)")

    save_per_class_bar(per_class, output_dir, f"Per-Class IoU (Mean: {best_val_iou:.4f})")
    save_training_plots(history, output_dir)
    save_history_to_file(history, best_val_iou, output_dir)

    # Also save last checkpoint
    torch.save(classifier.state_dict(),
               os.path.join(script_dir, "segmentation_head_last.pth"))

    print(f"\nDone. Best val mIoU = {best_val_iou:.4f}")
    print(f"Best checkpoint → segmentation_head_best.pth")
    print(f"Plots & metrics → {output_dir}/")


if __name__ == "__main__":
    main()
