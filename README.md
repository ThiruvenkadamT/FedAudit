# FedAudit

code and results for the paper

> T. Thiruvenkadam, N. Sivakumar, J. Emerson Raja , "FedAudit: Privacy-Preserving Auditing of Cross-Site Data Leakage and Shortcut Reliance in Federated Medical Imaging".

FedAudit is a four-stage audit that hospitals run before they report federated results. No plaintext
image, hash, embedding or explanation map leaves a hospital.

|Stage|What it does|What leaves a hospital|
|-|-|-|
|A|Finds exact and near-duplicate images across hospitals: SHA-256 hashes and SimHash band tokens of head-crop embeddings are matched with ECDH private set intersection, and candidate pairs are confirmed with a CKKS cosine-threshold test|PSI messages, encrypted embeddings of candidates, match bits|
|B|Measures accuracy inflation: the same model is scored on the full and the audited test set, also with class-standardized accuracy|accuracy figures|
|C|Shortcut audit: four bounded per-image statistics per predicted class, Gaussian noise calibrated with the analytic Gaussian mechanism, sums combined by secure aggregation|noised, masked sums|
|D|Shortcut mitigation in local training: text blanking or random text with a KL consistency loss; works with any aggregation rule|the usual model updates|

## Repository layout

```
fedaudit/
  data/          manifests and dataset builders, imaging utilities (head mask, text detector,
                 near-copy transformations), synthetic data for tests
  audit/         Stage A: embeddings, SimHash tokens, PSI, CKKS verification, central reference
  fl/            compact CNN, single-process simulator of FedAvg, FedProx, SCAFFOLD, FedNova,
                 FedDyn, FedAdam, MOON, FedBN, local-only and centralized training; evaluation
  shortcut/      Stage C (statistics, differential privacy, secure aggregation), deletion test
  analysis/      paired confidence intervals
  experiments/   dataset detection, audit experiments (E1-E3), training jobs (E4-E6),
                 two-GPU runner with resume, tables and figures
configs/experiments.json             all settings used in the paper
docs/EXPERIMENTS.md                  data, training, audit and experiment settings in detail
notebooks/FedAudit\\\\\\\\\\\\\\\_Experiments.ipynb Kaggle notebook with the package embedded (all experiments)
notebooks/FedAudit\\\\\\\\\\\\\\\_E4\\\\\\\\\\\\\\\_ResNet18.ipynb Kaggle notebook: brain MRI leakage inflation with ResNet-18
scripts/         CPU analyses: full-protocol brain MRI audit, cost experiment, information revealed
                 by the token exchange, agreement of the shortcut statistics with shortcut strength
paper/           scripts that produce the paper's numbers and figures; draw.io sources
tests/           fake dataset generator and an end-to-end smoke test
tools/           build the notebooks with the package (and the brain MRI splits) embedded
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10 or later. The PSI library is `openmined.psi` and the homomorphic encryption library is
`tenseal`.

## Data

All datasets are public. Attach them as Kaggle inputs, or place them under one folder and pass it as
`input\\\\\\\\\\\\\\\_root`; the code finds each dataset by its folder structure.

|Dataset|Source|
|-|-|
|Brain tumor MRI (Nickparvar)|https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset|
|Brain tumor classification (SARTAJ)|https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri|
|Br35H|https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection|
|figshare brain tumor dataset (Cheng et al., `.mat` files with patient IDs)|https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset|
|HAM10000|https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000|
|Brain tumor multimodal image (CT and MRI)|https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri|

The Kaggle copy of HAM10000 has no column naming each image's source. In that case the code draws four
sites at the lesion level with the sizes of the original sources. If a HAM10000 metadata file with a
`dataset` column (as distributed on Harvard Dataverse) is attached, the natural sources are used.

## Running the experiments

On Kaggle (two T4 GPUs):

1. Import `notebooks/FedAudit\\\\\\\\\\\\\\\_Experiments.ipynb`, attach the six datasets, enable the GPU and internet.
2. Run once with `QUICK\\\\\\\\\\\\\\\_TEST = True` (about 20 minutes) to check the setup.
3. Set `QUICK\\\\\\\\\\\\\\\_TEST = False` and commit. A session stops starting new runs after 11 hours. To continue,
add the notebook's own previous output as an input and commit again; finished runs are skipped.
4. Download `fedaudit\\\\\\\\\\\\\\\_results\\\\\\\\\\\\\\\_download.zip`.

The full study is 507 training runs plus the audit experiments, about three Kaggle sessions, and 30 further runs
with a pretrained ResNet-18 (`notebooks/FedAudit\\\\\\\\\\\\\\\_E4\\\\\\\\\\\\\\\_ResNet18.ipynb`, one session).

The brain MRI audit with the full protocol and the cost experiment run on a CPU:

```bash
python scripts/audit\\\\\\\\\\\\\\\_protocol\\\\\\\\\\\\\\\_brain.py path/to/fedaudit\\\\\\\\\\\\\\\_results
python scripts/cost\\\\\\\\\\\\\\\_scaling.py path/to/fedaudit\\\\\\\\\\\\\\\_results
python scripts/token\\\\\\\\\\\\\\\_leakage.py path/to/fedaudit\\\\\\\\\\\\\\\_results
python scripts/stagec\\\\\\\\\\\\\\\_ranking.py path/to/fedaudit\\\\\\\\\\\\\\\_results
python scripts/e4\\\\\\\\\\\\\\\_inflation.py path/to/fedaudit\\\\\\\\\\\\\\\_results   # also works on the ResNet-18 results folder
```

The ResNet-18 notebook is built from the results of the main experiments, so it uses the same splits
and flagged images: `python tools/build\\\\\\\\\\\\\\\_notebook\\\\\\\\\\\\\\\_backbone.py path/to/fedaudit\\\\\\\\\\\\\\\_results`.

Locally, without real data:

```bash
python tests/make\\\\\\\\\\\\\\\_fake\\\\\\\\\\\\\\\_kaggle\\\\\\\\\\\\\\\_input.py /tmp/fake 20
python tests/smoke\\\\\\\\\\\\\\\_test.py /tmp/fake /tmp/out
```

## Reproducing the paper's numbers and figures

The experiment outputs used in the paper (split lists, central reference, audit pairs, per-run
predictions and metrics, pixel embeddings; the ResNet-18 runs in `resnet18/`) are in `fedaudit\\\\\\\\\\\\\\\_results.zip`, attached to the release of
this repository. Unzip it and run:

```bash
python paper/collect\\\\\\\\\\\\\\\_numbers.py fedaudit\\\\\\\\\\\\\\\_results paper/numbers.json
python paper/make\\\\\\\\\\\\\\\_figures.py paper/numbers.json paper/figures
```

`make\\\\\\\\\\\\\\\_figures.py` writes each figure as a draw.io file together with 600-dpi PNG and TIFF images drawn
from the same shape list. The draw.io files in `paper/figures` can be edited directly.

## Experiments

|ID|Question|Data|
|-|-|-|
|E1|Accuracy of the private audit against a central reference (identical pixels, NCC >= 0.98 on 64 x 64 images, or dHash distance <= 2 with NCC >= 0.90), over encoders, band layouts and thresholds|brain MRI, HAM10000|
|E2|Time and traffic as images per site and number of sites grow|synthetic embeddings|
|E3|Recall on unaltered copies and copies altered by resizing, JPEG recompression, cropping and text overlay|brain MRI, HAM10000|
|E4|Accuracy inflation for ten training methods: natural leakage, image- vs lesion-level split, and injected copies at 0-10%|brain MRI, HAM10000|
|E5|Shortcut flag rate for epsilon in {0.5, 1, 2, 4} and without noise|CT and MRI with a planted marker, and with natural text|
|E6|Shortcut mitigations under FedAvg, SCAFFOLD and FedBN on five partitions|CT and MRI with a planted marker|

Settings are listed in `configs/experiments.json` and explained in `docs/EXPERIMENTS.md`.

## Notes on the implementation

* All parties are simulated in one process. Communication is measured as the size of the serialized
protocol messages.
* Band tokens that repeat within a site are sent through PSI once and mapped back to every image that
carries them.
* Above 3,000 candidate pairs per site pair, the CKKS test runs on a random sample of 3,000 pairs; all
decisions use the identical plaintext rule, and the full encrypted cost is extrapolated.
* Embeddings are centered with the federation mean, obtained by secure aggregation of per-site sums,
before hashing and verification.
* The feature-masking baseline zeroes the 10% of penultimate channels most correlated with the local
text indicator; the mask is refitted every five rounds and never shared.

## License

MIT. The datasets are not redistributed; see their own licenses.

