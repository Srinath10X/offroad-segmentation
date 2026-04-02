#set document(title: "TerraSeg: Adaptive Semantic Segmentation for Unstructured Environments", author: "Team Safaris")

#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.5cm, right: 2.5cm),
  numbering: "1",
  number-align: center,
)

#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.65em, first-line-indent: 1.5em)

// Section heading styling (Academic style)
#show heading.where(level: 1): it => {
  v(1.5em)
  text(size: 12pt, weight: "bold")[#it]
  v(0.5em)
}
#show heading.where(level: 2): it => {
  v(1em)
  text(size: 10pt, weight: "bold", style: "italic")[#it]
  v(0.5em)
}

// --- TITLE & AUTHORS ---
#align(center)[
  #v(1em)
  #text(size: 18pt, weight: "bold")[TerraSeg: Adaptive Semantic Segmentation for \ Unstructured Environments]
  #v(1.5em)
  #text(size: 11pt)[*Team Safaris*] \
  #v(0.5em)
  #text(size: 10pt)[Duality AI Hackathon · April 2026] \
  #text(size: 9pt, style: "italic")[Backbone: DINOv2 ViT-S/14 (frozen) $times$ Framework: PyTorch 2.x]
  #v(2.5em)
]

// --- ABSTRACT ---
#align(center)[
  #block(width: 85%, align(left)[
    #align(center)[*Abstract*]
    #v(0.5em)
    #set par(first-line-indent: 0pt)
    TerraSeg v3 achieves a *mean IoU of 0.49+* on the Duality AI offroad segmentation benchmark — absolutely crushing the target threshold of 0.2476 by nearly *2x*. This report details the architectural decisions, critical bug fixes, and training optimizations that drove this massive leap. The system pairs a frozen DINOv2 (ViT-S/14) backbone with a custom ConvNeXt-style segmentation head. Three targeted interventions — fixing a missing class in the label map, introducing class-weighted loss, and pre-caching backbone features — transformed a broken 0.2378 mIoU baseline into a leaderboard-dominating result without needing a bloated model architecture.
  ])
]

#v(2em)

= Introduction
Semantic segmentation in unstructured offroad environments presents unique challenges due to severe class imbalances and boundary ambiguity. Our target was to surpass the Duality AI benchmark threshold of 0.2476 mIoU. By shifting focus from architectural scaling to data-centric debugging and training optimization, TerraSeg v3 achieved an mIoU of 0.49+.

#align(center)[
  #table(
    columns: (1fr, 1fr, 1fr),
    fill: (_, row) => if row == 0 { rgb("#f1f5f9") } else { white },
    inset: 8pt,
    align: center,
    [*Metric*], [*Baseline (v2)*], [*TerraSeg v3*],
    [Mean IoU], [0.2378], [*0.4921*],
    [Pixel Accuracy], [0.8211], [*0.8015*],
    [Target (0.2476)], [Failed], [*Exceeded (2x)*],
  )
]

= Architectural Design

== Feature Extractor: DINOv2
We utilize Meta AI's *DINOv2* (Vision Transformer, ViT-S/14) as our primary feature extractor.
- *Universal Features:* Pre-trained on 142M images via self-supervised learning, it captures spatial relationships that traditional CNNs completely miss.
- *Frozen State:* The backbone is kept frozen to prevent catastrophic forgetting of universal visual primitives when exposed to synthetic textures.
- *Dimensionality:* The model produces 384-dimensional embeddings per $14 times 14$ pixel patch.
- *Feature Caching:* Since the backbone is frozen, features are extracted once and cached to disk — eliminating redundant recomputation and acting as a massive speed hack across all 25 training epochs.

== Segmentation Head: ConvNeXt-style (v3)
The v3 head doubles channel capacity from 128 to *256 channels* for improved rare-class discrimination:
1. *Large-Kernel Stem:* A 7x7 convolution (256ch) increases the receptive field early.
2. *Depthwise-Separable Block:* 256ch — reduces parameter count while maintaining non-linear complexity.
3. *Projection Block:* 256→128ch with 3x3 convolution for smooth dimensionality reduction.
4. *1x1 Classifier:* Maps to 11 output classes.
5. *Upsampling:* Bilinear interpolation restores the $1/14$ resolution back to the native input size ($476 times 266$).

= Dataset & Preprocessing

== Class Distribution Analysis
The dataset covers *11 semantic classes* from Duality AI's Falcon desert digital twin. Sky and Landscape represent >70% of total pixels, creating a severe long-tail distribution problem that directly suppresses mIoU for rare classes.

== Critical Bug: The "Invisible" Flowers Class
The original preprocessing pipeline had *no entry for pixel value 600 (Flowers)* in the label map. Every Flowers pixel was silently remapped to class 0 (Background), which:
- Corrupted Background IoU to near-zero.
- Injected toxic label noise into all spatially adjacent classes.
- Reduced effective training signal for 10% of the class vocabulary.

*The Fix:* A single line addition `600: 6` to the value map, with all subsequent classes renumbered to produce a clean $[0, 10]$ index space. One line of code; massive impact.

== Preprocessing Pipeline
Images are normalized using ImageNet statistics and resized to $476 times 266$ pixels — chosen to be exactly divisible by the DINOv2 patch size of 14, producing a clean $34 times 19$ token grid.

= Quantitative Performance

== Key Optimizations
To achieve the 0.49+ mIoU, we implemented five targeted interventions:

#table(
  columns: (0.8fr, 1.5fr, 1.2fr),
  fill: (_, row) => if row == 0 { rgb("#f8fafc") } else { white },
  inset: 8pt,
  [*Intervention*], [*Problem Context*], [*Empirical Result*],
  [Missing Flowers class], [600→Background label noise poisoned the dataset], [Unlocked correct class boundaries],
  [Weighted CrossEntropy], [Sky/Landscape dominated gradients; rare classes starved], [Rare class IoU lifted from ~0.01 to meaningful scores],
  [Feature caching], [DINOv2 recomputed identically every epoch (pure waste)], [Batch size 8→64; 25 epochs in ~25 min total],
  [Wider head (256ch)], [128ch head lacked capacity for 11-class discrimination], [Smoother class boundaries, higher per-class IoU],
  [Horizontal flip aug], [Token grid flipped correctly in (tokenH × tokenW) space], [Doubled effective dataset size for free],
)

== Performance Results
The impact of our data-centric fixes is immediately visible in the per-class IoU distribution. The baseline model failed entirely on rare classes, whereas TerraSeg v3 recovers them.

#align(center)[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    figure(
      image("../.github/assets/iou_mean.png", width: 100%),
      caption: [*Baseline (v2)*: Rare classes suppressed (mIoU 0.2378)]
    ),
    figure(
      image("../.github/assets/optimized_bar.png", width: 100%),
      caption: [*TerraSeg v3*: Long-tail classes unlocked (mIoU 0.49+)]
    )
  )
]

= Qualitative Evaluation

We didn't just beat the metrics; the visual recovery is undeniable. Below is the head-to-head proof of why data-centric debugging works. In the previous version, the model was practically blind to the Flowers class and bled noisy background predictions everywhere. In v3, the semantics are razor-sharp.

#align(center)[
  #figure(
    image("../.github/assets/optimized_class.png", width: 90%),
    caption: [*Baseline (v2)*: Blind to flowers, messy class boundaries, heavy label noise.]
  )
  #v(1em)
  #figure(
    image("../.github/assets/sample.png", width: 90%),
    caption: [*TerraSeg v3 (Ours)*: Flowers correctly detected, razor-sharp boundaries, clean semantics.]
  )
]

== Success Cases & Failure Modes
- *Sky Segmentation:* Near-perfect horizon separation (IoU > 0.94).
- *Landscape Continuity:* Large ground regions segmented with high confidence.
- *The "Invisible" Flowers:* Previously completely ignored by the model; now successfully identified after the label fix.
- *Boundary Ambiguity (Failure Mode):* Dry Grass vs. Landscape confusion at low-contrast transition zones.
- *Occlusion (Failure Mode):* Small obstacles (Logs, Rocks) occasionally absorbed into surrounding ground class.

= Conclusion
TerraSeg v3 proves that *data-centric debugging and optimized pipelines* completely outclass blind architectural scaling. The jump from 0.2378 to 0.49+ mIoU came directly from correctly labeling the 11 classes, forcing the model to care about rare pixels with weighted loss, and nuking redundant compute with feature caching. The frozen DINOv2 backbone provides strong universal features that transfer effortlessly to synthetic desert imagery, proving we built a robust, highly adaptive segmentation system.
