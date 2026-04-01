"""
Segmentation Test/Inference Script - TerraSeg v3 (Fixed)
Matches train_v3_fast.py exactly:
  - 11 classes including Flowers (600 -> 6)
  - SegmentationHeadConvNeXt with 256 channels
  - Correct image size (476, 266)
  - Correct mask loading (no * 255 bug)
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

# ============================================================================
# Class config  —  must match train_v3_fast.py exactly
# ============================================================================

value_map = {
    0:     0,   # Background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers        ← was MISSING in original test.py
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}
n_classes = len(value_map)   # 11

CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# 11 colors — one per class
COLOR_PALETTE = np.array([
    [0,   0,   0  ],  # Background    — black
    [34,  139, 34 ],  # Trees         — forest green
    [0,   255, 0  ],  # Lush Bushes   — lime
    [210, 180, 140],  # Dry Grass     — tan
    [139, 90,  43 ],  # Dry Bushes    — brown
    [128, 128, 0  ],  # Ground Clutter— olive
    [255, 105, 180],  # Flowers       — hot pink
    [139, 69,  19 ],  # Logs          — saddle brown
    [128, 128, 128],  # Rocks         — gray
    [160, 82,  45 ],  # Landscape     — sienna
    [135, 206, 235],  # Sky           — sky blue
], dtype=np.uint8)


def convert_mask(mask):
    arr = np.array(mask, dtype=np.int32)
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        out[arr == raw] = new
    return Image.fromarray(out)


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(n_classes):
        color_mask[mask == c] = COLOR_PALETTE[c]
    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    """Works with val set (has masks) or test set (no masks — pass masks_dir=None)."""
    def __init__(self, data_dir, W=476, H=266, has_masks=True):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.has_masks = has_masks
        if has_masks:
            self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = sorted(os.listdir(self.image_dir))
        self.W, self.H = W, H

        self.img_tf = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.data_ids)

    def __getitem__(self, idx):
        fid   = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, fid)).convert("RGB")
        image = self.img_tf(image)

        if self.has_masks:
            mask = convert_mask(Image.open(os.path.join(self.masks_dir, fid)))
            mask = torch.from_numpy(
                np.array(mask.resize((self.W, self.H), Image.NEAREST))
            ).long()
        else:
            mask = torch.zeros(self.H, self.W, dtype=torch.long)  # dummy

        return image, mask, fid


# ============================================================================
# Model  —  must match train_v3_fast.py exactly (256 channels)
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

def compute_iou(pred_logits, target, img_hw):
    preds    = F.interpolate(pred_logits, size=img_hw,
                             mode="bilinear", align_corners=False)
    pred_cls = torch.argmax(preds, dim=1).view(-1)
    tgt      = target.view(-1)
    per_class = []
    for c in range(n_classes):
        p = pred_cls == c; t = tgt == c
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        per_class.append(float('nan') if union == 0 else (inter / union).item())
    return float(np.nanmean(per_class)), per_class


def compute_pixel_accuracy(pred_logits, target, img_hw):
    preds = F.interpolate(pred_logits, size=img_hw,
                          mode="bilinear", align_corners=False)
    return (torch.argmax(preds, dim=1) == target).float().mean().item()


# ============================================================================
# Visualization
# ============================================================================

def save_comparison(img_tensor, gt_mask, pred_mask, out_path, title=""):
    img  = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.clip(np.moveaxis(img, 0, -1) * std + mean, 0, 1)

    gt_color   = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);       axes[0].set_title('Input Image');  axes[0].axis('off')
    axes[1].imshow(gt_color);  axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pred_color);axes[2].set_title('Prediction');   axes[2].axis('off')
    plt.suptitle(title); plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


def save_metrics_summary(mean_iou, class_iou, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Text file
    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write("TERRASEG V3 EVALUATION RESULTS\n" + "="*50 + "\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write("="*50 + "\n\nPer-Class IoU:\n" + "-"*40 + "\n")
        for name, iou in zip(CLASS_NAMES, class_iou):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A (not in dataset)"
            f.write(f"  {name:<20}: {iou_str}\n")

    # Bar chart
    colors = [COLOR_PALETTE[i] / 255.0 for i in range(n_classes)]
    valid  = [iou if not np.isnan(iou) else 0 for iou in class_iou]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(CLASS_NAMES, valid, color=colors, edgecolor='black')
    ax.axhline(mean_iou, color='red', linestyle='--', label=f'Mean {mean_iou:.4f}')
    ax.set_title(f'Per-Class IoU (Mean: {mean_iou:.4f})')
    ax.set_ylabel('IoU'); ax.set_ylim(0, 1); ax.legend()
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_iou.png'), dpi=150); plt.close()

    print(f"\nMetrics saved to {output_dir}/")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'segmentation_head_best.pth'))
    parser.add_argument('--data_dir',   type=str,
                        default=os.path.join(script_dir, '..', 'data',
                                             'Offroad_Segmentation_Training_Dataset', 'val'))
    parser.add_argument('--output_dir', type=str, default='./predictions')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of side-by-side comparison images to save')
    parser.add_argument('--no_masks', action='store_true',
                        help='Set if test set has no ground truth masks')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    W, H = 476, 266   # must match training

    # ── Dataset ──────────────────────────────────────────────────────────────
    has_masks = not args.no_masks
    print(f"Loading dataset from {args.data_dir} (has_masks={has_masks})...")
    dataset = MaskDataset(args.data_dir, W=W, H=H, has_masks=has_masks)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=2, pin_memory=True)
    print(f"Loaded {len(dataset)} samples.")

    # ── Backbone ─────────────────────────────────────────────────────────────
    print("Loading DINOv2 backbone (dinov2_vits14)...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Probe embedding dim
    with torch.no_grad():
        sample, _, _ = dataset[0]
        feat  = backbone.forward_features(sample.unsqueeze(0).to(device))["x_norm_patchtokens"]
    n_emb = feat.shape[2]
    print(f"Embedding dim: {n_emb}")

    # ── Classifier ───────────────────────────────────────────────────────────
    print(f"Loading weights from {args.model_path}...")
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_emb,
        out_channels=n_classes,
        tokenW=W // 14,
        tokenH=H // 14,
    ).to(device)
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier.eval()
    print("Model loaded.\n")

    # ── Output dirs ──────────────────────────────────────────────────────────
    masks_dir      = os.path.join(args.output_dir, 'masks')
    masks_color_dir= os.path.join(args.output_dir, 'masks_color')
    comparisons_dir= os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Inference loop ────────────────────────────────────────────────────────
    iou_scores, pixel_accs, all_class_iou = [], [], []
    sample_count = 0
    img_hw = (H, W)

    print(f"Running inference on {len(dataset)} images...")
    with torch.no_grad():
        pbar = tqdm(loader, desc="Inference", unit="batch")
        for imgs, masks, fids in pbar:
            imgs, masks = imgs.to(device), masks.to(device)

            feat   = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(feat)

            pred_masks = torch.argmax(
                F.interpolate(logits, size=img_hw, mode="bilinear", align_corners=False),
                dim=1
            )

            if has_masks:
                miou, class_iou = compute_iou(logits, masks, img_hw)
                acc = compute_pixel_accuracy(logits, masks, img_hw)
                iou_scores.append(miou)
                pixel_accs.append(acc)
                all_class_iou.append(class_iou)
                pbar.set_postfix(iou=f"{miou:.3f}")

            # Save outputs per image
            for i in range(imgs.shape[0]):
                fid       = fids[i]
                base_name = os.path.splitext(fid)[0]
                pred_np   = pred_masks[i].cpu().numpy().astype(np.uint8)

                # Raw class-ID mask
                Image.fromarray(pred_np).save(
                    os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Coloured mask
                cv2.imwrite(
                    os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                    cv2.cvtColor(mask_to_color(pred_np), cv2.COLOR_RGB2BGR))

                # Comparison (first N samples, only if we have GT)
                if has_masks and sample_count < args.num_samples:
                    save_comparison(
                        imgs[i], masks[i], pred_masks[i],
                        os.path.join(comparisons_dir,
                                     f'sample_{sample_count:04d}_comparison.png'),
                        title=fid,
                    )
                sample_count += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    if has_masks:
        mean_iou      = float(np.nanmean(iou_scores))
        mean_acc      = float(np.mean(pixel_accs))
        avg_class_iou = list(np.nanmean(all_class_iou, axis=0))

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Mean IoU      : {mean_iou:.4f}")
        print(f"Pixel Accuracy: {mean_acc:.4f}")
        print("-"*50)
        print("Per-Class IoU:")
        for name, iou in zip(CLASS_NAMES, avg_class_iou):
            bar = "█" * int((iou if not np.isnan(iou) else 0) * 30)
            print(f"  {name:<20} {iou:.4f}  {bar}")
        print("="*50)

        save_metrics_summary(mean_iou, avg_class_iou, args.output_dir)
    else:
        print(f"\nInference complete — no GT masks, skipping metrics.")

    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  masks/         — raw class-ID PNGs")
    print(f"  masks_color/   — coloured RGB PNGs")
    if has_masks:
        print(f"  comparisons/   — side-by-side comparisons ({args.num_samples} samples)")
    print(f"  evaluation_metrics.txt + per_class_iou.png")


if __name__ == "__main__":
    main()
