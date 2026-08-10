# Project README 

## Core Features

- **Backbone**: `Asmamba` with **SA‑SSM** (Snake‑Adaptive SSM: snake‑shaped bidirectional scanning + adaptive gating)
- **Decode Head**: `UPerHeadDictEnhanced` (enhanced dynamic dictionary: relation matrix + contrastive loss + adaptive weight)
- **Sampler / Hook**: `IterBasedDCBSComplementSampler` + `DCBSComplementHook` (complementary sample detection)
- **Losses**: Multi‑loss combination of Dice + Focal + Lovasz + Tversky
- **Datasets**: `CustomThreeClassDataset` / `CustomThreeClassOversampleDataset`
- **Test set**: Test data is stored under `data_test/`, containing `img` and `mask` sub‑folders

## Directory Structure

```
Asmamba_sa_ssm_upernet_dict_enhanced_project/
├── configs/
│   └── Asmamba_sa_ssm_upernet_dict_enhanced.py   # Self‑contained training config, ready‑to‑run
├── mmseg/                                      # mmseg framework with all custom modules for this experiment
│   ├── models/
│   │   ├── backbones/    # Asmamba / SA‑SSM / 4D‑8D‑SSM, etc.
│   │   ├── decode_heads/ # UPerHeadDictEnhanced / UPerHeadDict
│   │   ├── losses/       # Dice / Focal / Lovasz / Tversky
│   │   └── utils/        # dynamic_dictionary(_enhanced) / adaptive_loss_weights
│   ├── datasets/         # custom_three_class(_oversample) + samplers/dcbs_*
│   ├── engine/hooks/     # dcbs_complement_hook / csv_logger_hook
│   ├── evaluation/  structures/  utils/  visualization/  registry/
├── tools/
│   ├── train.py          # Training entry script
│   └── test.py           # Test / evaluation entry script
├── data/                 # Placeholder for training & validation data (populate with real data, see below)
│   ├── img/{train,val}/
│   └── mask/{train,val}/
├── data_test/            # Test dataset directory
│   ├── img/test/
│   └── mask/test/
├── requirements/         # Dependency lists
├── requirements.txt
├── setup.py / setup.cfg / MANIFEST.in
├── run.sh                # One‑click training script
└── README.md
```

## Environment Setup

```
conda activate Asmamba          
pip install -e .                # Optional: install mmseg in editable mode
```

Dependency versions:

| Dependency | Version                                           |
| ---------- | ------------------------------------------------- |
| Python     | 3.9                                               |
| PyTorch    | 2.1.2+cu121                                       |
| mmcv       | 2.1.0                                             |
| mmengine   | 0.10.7                                            |
| timm       | ≥0.9 (provides DropPath/to_2tuple/register_model) |
| mamba_ssm  | ≥1.0 (provides native `Mamba` operator)           |
| einops     | ≥0.6                                              |

> The `Asmamba` backbone requires CUDA‑compiled `mamba_ssm` and `timm`. Missing packages will trigger import errors. Additional runtime dependencies are listed in `requirements/runtime.txt` and `requirements/mminstall.txt`.

## Data Preparation

Training and validation data root is set to `data/`. Place your real training‑validation dataset inside this folder:

```
data/
├── img/
│   ├── train/   *.png
│   └── val/     *.png
└── mask/
    ├── train/   *.png (file names match corresponding images)
    └── val/     *.png
```

Test dataset is located under `data_test/`:

```
data_test/
├── img/
│   └── test/    *.png
└── mask/
    └── test/    *.png (file names match corresponding test images)
```

Original source dataset resides at `/home/zjl/envs/Samba-main/data/` in the original environment. You can copy files directly:

```
# Copy train‑val data
cp -r /home/zjl/envs/Samba-main/data/img/* data/img/
cp -r /home/zjl/envs/Samba-main/data/mask/* data/mask/

# Copy test data to data_test
cp -r /home/zjl/envs/Samba-main/data/img/test/* data_test/img/test/
cp -r /home/zjl/envs/Samba-main/data/mask/test/* data_test/mask/test/
```

## Training

```
# One‑click script
bash run.sh

# Manual execution
python tools/train.py configs/Asmamba_sa_ssm_upernet_dict_enhanced.py
```

- Default setting: single‑GPU training for 80000 iterations. Outputs are saved to `output/Asmamba_sa_ssm_upernet_dict_enhanced/`.
- Specify custom output directory: `python tools/train.py <config> --work-dir <dir>`
- Resume from checkpoint: `python tools/train.py <config> --work-dir <dir> --resume`

## Test / Evaluation

```
python tools/test.py \
    configs/Asmamba_sa_ssm_upernet_dict_enhanced.py \
    output/Asmamba_sa_ssm_upernet_dict_enhanced/best_mIoU_iter_78000.pth
```

Metrics including mIoU, mDice and mFscore will be printed. Segmentation visualizations will also be dumped.

## Notes

- `configs/Asmamba_sa_ssm_upernet_dict_enhanced.py` is the self‑contained training configuration dumped by mmengine, without relying on `_base_` config inheritance chains.
- Trained weights (`.pth`, ~13 GB) are runtime outputs and are not included in this repository. Copy weight files from the original `output/` directory when needed.
- The model is trained from scratch (`load_from = None` in config), no external pre‑trained weights are required.
