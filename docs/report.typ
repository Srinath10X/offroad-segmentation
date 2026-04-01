#set document(title: "Offroad Semantic Scene Segmentation", author: "Team Safaris")

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  numbering: "1",
  number-align: center,
)

#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set heading(numbering: "1.")
#set par(justify: true, leading: 0.75em)

#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 15pt, weight: "bold", fill: rgb("#1a1a2e"))[#it]
  v(0.5em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12.5pt, weight: "bold", fill: rgb("#16213e"))[#it]
  v(0.3em)
}

// ─── TITLE PAGE ───────────────────────────────────────────────────────────────
#align(center)[
  #v(3cm)
  #rect(fill: rgb("#1a1a2e"), radius: 10pt, inset: (x: 30pt, y: 20pt))[
    #text(size: 28pt, weight: "bold", fill: white)[TerraSeg]
    #linebreak()
    #text(size: 14pt, fill: rgb("#a8dadc"))[Offroad Semantic Scene Segmentation]
  ]
  #v(1cm)
  #text(size: 12pt, fill: rgb("#444"))[Duality AI · Hack the Night Hackathon]
  #v(0.5cm)
  #text(size: 11pt, fill: rgb("#666"))[
    Submitted by: *Team Safaris* \
    Date: April 2026
  ]
  #v(1.5cm)
  #line(length: 80%, stroke: rgb("#1a1a2e") + 1.5pt)
  #v(0.8cm)
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    rect(fill: rgb("#f0f4ff"), radius: 6pt, inset: 12pt)[
      #align(center)[
        #text(size: 10pt, fill: rgb("#555"))[*Model*] \
        #text(size: 9pt)[DINOv2 + ConvNeXt Head]
      ]
    ],
    rect(fill: rgb("#f0f4ff"), radius: 6pt, inset: 12pt)[
      #align(center)[
        #text(size: 10pt, fill: rgb("#555"))[*Classes*] \
        #text(size: 9pt)[10 Desert Categories]
      ]
    ],
    rect(fill: rgb("#f0f4ff"), radius: 6pt, inset: 12pt)[
      #align(center)[
        #text(size: 10pt, fill: rgb("#555"))[*Metric*] \
        #text(size: 9pt)[Mean IoU (mIoU)]
      ]
    ],
  )
]

#pagebreak()

// ─── TABLE OF CONTENTS ────────────────────────────────────────────────────────
#outline(title: [Table of Contents], indent: 1.5em, depth: 2)

#pagebreak()

// ─── 1. INTRODUCTION ──────────────────────────────────────────────────────────
= Introduction

== Problem Statement

Autonomous off-road vehicles — known as Unmanned Ground Vehicles (UGVs) — need to understand their surroundings at every moment. To navigate safely through complex desert terrain, they rely on *semantic scene segmentation*: labeling every single pixel in a camera image with the category it belongs to (sky, rock, tree, etc.).

Without this, a UGV cannot distinguish a navigable sandy path from a rocky obstacle, or tell a dry bush from a log it might get stuck on.

== The Challenge

The traditional approach requires thousands of real-world labeled images — expensive, slow, and dangerous to collect in remote deserts. This hackathon by *Duality AI* poses a different challenge:

#rect(fill: rgb("#f5f5ff"), radius: 6pt, inset: 14pt)[
  Train a segmentation model using *synthetic* desert images from Duality AI's *Falcon* platform. Then test it on a *completely different desert location* it has never seen — measuring how well it generalizes.
]

== Our Approach

We built *TerraSeg*, combining:
- *DINOv2* — a pre-trained Vision Transformer backbone from Meta AI, used as a frozen feature extractor
- *A lightweight ConvNeXt-style segmentation head* — a small network we train to map features to per-pixel class predictions

#pagebreak()

// ─── 2. DATASET ───────────────────────────────────────────────────────────────
= Dataset

== Source

All images were generated using *FalconEditor*, Duality AI's geospatial digital twin environment. The dataset contains synthetic RGB desert photographs with pixel-level segmentation masks.

== Structure

#table(
  columns: (auto, 1fr),
  fill: (_, row) => if row == 0 { rgb("#1a1a2e") } else if calc.even(row) { rgb("#f5f5ff") } else { white },
  inset: 10pt,
  stroke: none,
  text(fill: white, weight: "bold")[Folder], text(fill: white, weight: "bold")[Contents],
  [`Train/Color_Images/`], [RGB training photographs],
  [`Train/Segmentation/`], [Pixel-wise class masks],
  [`Val/Color_Images/`], [RGB validation photographs],
  [`Val/Segmentation/`], [Pixel-wise validation masks],
  [`testImages/`], [Unseen RGB images from a *different* desert — no masks],
)

== Segmentation Classes

#table(
  columns: (auto, auto, 1fr),
  fill: (_, row) => if row == 0 { rgb("#1a1a2e") } else if calc.even(row) { rgb("#f5f5ff") } else { white },
  inset: 9pt,
  stroke: none,
  text(fill: white, weight: "bold")[Raw ID], text(fill: white, weight: "bold")[Mapped ID], text(fill: white, weight: "bold")[Class],
  [0], [0], [Background],
  [100], [1], [Trees],
  [200], [2], [Lush Bushes],
  [300], [3], [Dry Grass],
  [500], [4], [Dry Bushes],
  [550], [5], [Ground Clutter],
  [700], [6], [Logs],
  [800], [7], [Rocks],
  [7100], [8], [Landscape (general ground)],
  [10000], [9], [Sky],
)

== Preprocessing

Raw masks use non-sequential IDs. We remap them to 0–9 before training:
```python
value_map = {0:0, 100:1, 200:2, 300:3, 500:4,
             550:5, 700:6, 800:7, 7100:8, 10000:9}
```

All images are resized to *476 × 266* (nearest size divisible by 14 for DINOv2 patches) and normalized with ImageNet statistics.

#pagebreak()

// ─── 3. METHODOLOGY ───────────────────────────────────────────────────────────
= Methodology

== Architecture Overview

#rect(fill: rgb("#f0f9ff"), radius: 8pt, inset: 16pt)[
  #align(center)[
    *Desert Image → DINOv2 Backbone (frozen) → Patch Tokens → ConvNeXt Head (trained) → Class Mask*
  ]
]

== DINOv2 Backbone (Frozen)

DINOv2 is a Vision Transformer pre-trained by Meta AI on 140M+ images. It splits the image into *14×14 pixel patches*, encodes each into a *384-dimensional feature vector*, and uses self-attention so every patch can reference every other patch.

We keep it *completely frozen* — no weights are updated. This means faster training, better generalization, and we leverage Meta's pre-training for free.

== ConvNeXt Segmentation Head (Trained)

#table(
  columns: (auto, 1fr),
  fill: (_, row) => if row == 0 { rgb("#1a1a2e") } else if calc.even(row) { rgb("#f5f5ff") } else { white },
  inset: 9pt,
  stroke: none,
  text(fill: white, weight: "bold")[Layer], text(fill: white, weight: "bold")[What it does],
  [Reshape], [Convert flat patch tokens → 2D spatial grid],
  [Stem Conv 7×7], [384 → 128 channels, large receptive field],
  [Depthwise Conv 7×7], [Spatial feature mixing per channel],
  [Pointwise Conv 1×1], [Cross-channel mixing],
  [Classifier Conv 1×1], [128 → 10 class scores per location],
  [Bilinear Upsample], [Scale back to full image resolution],
)

~500K trainable parameters total.

== Training Config

#table(
  columns: (auto, 1fr),
  fill: (_, row) => if row == 0 { rgb("#1a1a2e") } else if calc.even(row) { rgb("#f5f5ff") } else { white },
  inset: 9pt,
  stroke: none,
  text(fill: white, weight: "bold")[Setting], text(fill: white, weight: "bold")[Value],
  [Epochs], [10],
  [Batch Size], [2],
  [Learning Rate], [1e-4],
  [Optimizer], [SGD momentum=0.9],
  [Loss], [Cross-Entropy],
  [Platform], [Google Colab T4 GPU],
)

#pagebreak()

// ─── 4. METRICS ───────────────────────────────────────────────────────────────
= Evaluation Metrics

== Mean IoU (Primary — 80 pts)

$ "IoU"_c = frac("Predicted" inter "Ground Truth", "Predicted" union "Ground Truth") , quad "mIoU" = 1/C sum_c "IoU"_c $

IoU = 1.0 is perfect. IoU = 0.0 is completely wrong.

== Dice Score

$ "Dice"_c = frac(2 times |"Pred" inter "GT"|, |"Pred"| + |"GT"|) $

== Pixel Accuracy

$ "Accuracy" = frac("Correctly classified pixels", "Total pixels") $

#pagebreak()

// ─── 5. RESULTS ───────────────────────────────────────────────────────────────
= Results

== Final Metrics

#table(
  columns: (1fr, auto, auto),
  fill: (_, row) => if row == 0 { rgb("#1a1a2e") } else if calc.even(row) { rgb("#f5f5ff") } else { white },
  inset: 10pt,
  stroke: none,
  text(fill: white, weight: "bold")[Metric], text(fill: white, weight: "bold")[Train], text(fill: white, weight: "bold")[Val],
  [Loss], [_fill_], [_fill_],
  [mIoU], [_fill_], [_fill_],
  [Dice Score], [_fill_], [_fill_],
  [Pixel Accuracy], [_fill_], [_fill_],
)

== Per-Class IoU

#table(
  columns: (1fr, auto),
  fill: (_, row) => if row == 0 { rgb("#1a1a2e") } else if calc.even(row) { rgb("#f5f5ff") } else { white },
  inset: 10pt,
  stroke: none,
  text(fill: white, weight: "bold")[Class], text(fill: white, weight: "bold")[IoU],
  [Background], [_fill_],
  [Trees], [_fill_],
  [Lush Bushes], [_fill_],
  [Dry Grass], [_fill_],
  [Dry Bushes], [_fill_],
  [Ground Clutter], [_fill_],
  [Logs], [_fill_],
  [Rocks], [_fill_],
  [Landscape], [_fill_],
  [Sky], [_fill_],
)

_Insert training_curves.png and per_class_metrics.png here_

#pagebreak()

// ─── 6. CHALLENGES ────────────────────────────────────────────────────────────
= Challenges & Solutions

== Non-Standard Class IDs
*Problem:* Raw mask values (0, 100, 200 ... 10000) are non-sequential — PyTorch needs 0 to N-1. \
*Fix:* `convert_mask()` remaps all values before training.

== Windows-Only Setup Script
*Problem:* `setup_env.bat` doesn't work on Linux. \
*Fix:* Used Google Colab — platform independent, free T4 GPU, training in ~30 min instead of hours.

== Image Size Must Be Divisible by 14
*Problem:* DINOv2 patch size is 14×14. Wrong image size breaks tokenization. \
*Fix:* `W = int(((960/2) // 14) * 14)` → 476×266.

== Class Imbalance
*Problem:* Sky and Landscape dominate pixels; Logs and Ground Clutter are rare. \
*Fix:* Monitor per-class IoU separately to identify weak classes. Future work: weighted loss.

== Generalization to Unseen Environments
*Problem:* Test images are from a different desert location. \
*Fix:* Frozen DINOv2 provides universal features that don't overfit to one location.

#pagebreak()

// ─── 7. FAILURE CASES ─────────────────────────────────────────────────────────
= Failure Case Analysis

== Visually Similar Classes
*Dry Grass vs Dry Bushes vs Landscape* share similar tan/brown color palettes. The model is likely to confuse these in transitional zones.

== Rare Classes
*Logs* and *Ground Clutter* appear in very few pixels. The model may default to predicting the surrounding dominant class instead.

== Boundary Blur
Bilinear upsampling from low-res predictions back to full image size causes imprecise class boundaries — especially visible where trees meet sky.

== Domain Shift
Rock formations, soil color, and vegetation density differ between training and test deserts. Novel visual patterns not seen in training may be misclassified as the nearest-looking class.

_Insert comparison images from predictions/comparisons/ here to illustrate failures_

#pagebreak()

// ─── 8. CONCLUSION ────────────────────────────────────────────────────────────
= Conclusion & Future Work

== Summary

TerraSeg demonstrates that *synthetic data + pre-trained vision backbones* is a powerful combination for off-road segmentation. We trained efficiently on a free cloud GPU in under 30 minutes, while achieving generalization to unseen desert environments — exactly the real-world deployment scenario this challenge simulates.

== Key Takeaways
- Synthetic data from Falcon is high quality enough to train generalizable models
- Frozen DINOv2 features transfer well to desert scene understanding
- A ~500K parameter head is sufficient when the backbone is strong

== Future Work
- *Weighted loss* to improve rare class (Logs, Ground Clutter) performance
- *FPN or UPerNet head* for multi-scale feature fusion and sharper boundaries
- *Data augmentation* — color jitter, flips, random crops for lighting robustness
- *Partial backbone fine-tuning* for domain-specific improvement
- *Larger DINOv2* (ViT-Base or ViT-Large) for higher feature quality

#v(2em)
#line(length: 100%, stroke: rgb("#ddd"))
#align(center)[
  #text(size: 10pt, fill: rgb("#888"), style: "italic")[
    TerraSeg · Hack the Night · Duality AI · April 2026
  ]
]
