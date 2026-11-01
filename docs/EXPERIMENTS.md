# Experimental settings

All settings used for the paper are collected in `configs/experiments.json`. This page explains them.

## Data

| Federation | Sites | Images | Classes | Split |
|---|---|---|---|---|
| Brain MRI | figshare (Cheng et al.), SARTAJ, Br35H, Nickparvar | 16,528 (15,028 used for training and testing) | glioma, meningioma, pituitary tumor, no tumor | published splits of SARTAJ and Nickparvar with 1/8 of training held out for validation; figshare and Br35H 70/10/20 per class |
| HAM10000 | 4 sites drawn at the lesion level with the sizes of the original sources | 10,015 (7,470 lesions) | 7 | image-level and lesion-level 70/10/20 |
| CT and MRI | 5 simulated sites | 7,955 after removing copies (identical pixels or NCC >= 0.99) | healthy, tumor | iid; Dirichlet label skew with alpha 1.0, 0.5, 0.1; modality skew (3 sites 80% CT, 2 sites 80% MRI) |

The 1,500 Br35H tumor images have no subtype label; they take part in the audit only. In the CT and
MRI federation a class-linked text marker is planted at sites 0 and 1, in the top-right corner, naming
the true class with probability 0.9.

## Training

- Compact CNN: four blocks of two 3x3 convolutions with batch normalization and ReLU, max pooling,
  32/64/128/256 channels, global average pooling, dropout 0.3, linear classifier (1.17 M parameters).
- 128 x 128 inputs, grayscale for MRI and CT, RGB for dermoscopy; horizontal flip with p = 0.5.
- 30 rounds, 1 local epoch, batch 32, SGD with lr 0.01, momentum 0.9 (no local momentum for SCAFFOLD
  and FedNova), weight decay 1e-4.
- FedProx mu 0.01; FedDyn alpha 0.01; FedAdam server lr 0.01, betas 0.9/0.99, tau 1e-3;
  MOON mu 1, temperature 0.5.
- Seeds 0, 1, 2. With fixed splits the seed changes initialization and batch order; in the controlled
  leakage and CT/MRI experiments it also changes the partition, injected copies and planted markers.
- Each test image is scored by the model its own site deploys: the global model, or the personalized
  model for FedBN and local-only training.

## Stage A (duplicate audit)

- Exact step: SHA-256 of the decoded pixel array, matched with ECDH PSI in both directions.
- Near-duplicate step: head crop, frozen encoder, centering with the federation mean, SimHash with
  n bands of w bits, tokens = first 128 bits of SHA-256(band index, band bits), ECDH PSI on tokens,
  candidate pairs share at least one token.
- Verification: CKKS (polynomial degree 8192, scale 2^40), the partner returns
  Enc(r (cos - tau)) with r uniform in [1, 64]; only the sign is kept.
- Above 3,000 candidates per site pair the encrypted test runs on a random sample of 3,000; decisions
  use the identical plaintext rule and the full encrypted cost is extrapolated.
- Sweep: 3 encoders x 6 band layouts x 8 thresholds = 144 configurations per dataset.

## Stage B (inflation)

Inflation is Acc(D) - Acc(D without L). The class-standardized version averages per-class accuracy
with the class shares of D and leaves out classes with fewer than 50 test images after removal.

## Stage C (shortcut audit)

128 test images per site; per image: text present outside the head, outside-head share of SmoothGrad
attribution (8 samples), absolute change of the predicted-class probability when text is blanked and
when text of another class is pasted. Per predicted class the sums, the sums of squares of the two
sensitivities and the count form a 7-entry vector (L2 sensitivity sqrt(7) under add/remove adjacency).
Noise from the analytic Gaussian mechanism with delta 1e-5, split across sites, summed with
pairwise-masked secure aggregation. The reference model (head-cropped) is released the same way.
Flag rule: outside-head attribution above the reference + 0.05, and masking or swap sensitivity
significantly above 0.05 (one-sided z-test at 0.05). Twenty noise draws per run and epsilon.

## Stage D and baselines

Stage D: with p = 0.5 an image is replaced by a copy with its text blanked or random text pasted;
loss = CE + lambda KL(f(x) || f(T(x))), lambda = 1. Baselines: head cropping at every site, blanking
text in training images, and local feature masking of the 10% penultimate channels most correlated
with the site's text indicator (refitted every five rounds, never shared).

## Experiments

| ID | Runs | Content |
|---|---|---|
| E1 | audit only | precision and recall of all 144 configurations against the central reference |
| E2 | audit only | time and traffic for 2 sites x {1k, 5k, 20k} images and {3, 4, 5, 8, 10} sites x 2k images (synthetic 2,048-d embeddings, 2% planted duplicates) |
| E3 | audit only | recall on 300 copies per transformation (unaltered, resize, JPEG, crop, text overlay) |
| E4 | 30 + 60 + 180 | brain MRI (10 methods x 3 seeds); HAM10000 image vs lesion split (10 x 2 x 3); controlled leakage (4 methods x 3 partitions x 5 rates x 3 seeds) |
| E5/E6 | 225 + 12 | CT and MRI with the planted marker: 3 algorithms x 5 partitions x 5 variants x 3 seeds; without the marker (natural text): FedAvg, iid, 4 variants x 3 seeds. E5 applies Stage C to these models |

Paired differences are taken over seeds (and partitions and algorithms where pooled); intervals are
95% Student-t intervals with two-sided paired t-test p values.

## Hardware

Training runs: Kaggle, two NVIDIA T4 GPUs (507 runs in three sessions, plus 30 ResNet-18 runs in one session). The full-protocol brain audit
and the cost experiment (`scripts/`): two-core 2.1 GHz CPU, 8 GB memory.
