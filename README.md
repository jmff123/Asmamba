
## 核心特性

- **Backbone**：`Asmamba`，启用 **SA-SSM**（Snake-Adaptive SSM 蛇形双向扫描 + 自适应门控）
- **Decode Head**：`UPerHeadDictEnhanced`（增强动态字典：关系矩阵 + 对比损失 + 自适应权重）
- **Sampler / Hook**：`IterBasedDCBSComplementSampler` + `DCBSComplementHook`（互补样本检测）
- **Losses**：Dice + Focal + Lovasz + Tversky 多损失组合
- **数据**：`CustomThreeClassDataset` / `CustomThreeClassOversampleDataset`

## 目录结构

```
Asmamba_sa_ssm_upernet_dict_enhanced_project/
├── configs/
│   └── Asmamba_sa_ssm_upernet_dict_enhanced.py   # 训练配置（自包含，可直接运行）
├── mmseg/                                      # mmseg 框架包（含本实验全部自定义模块）
│   ├── models/
│   │   ├── backbones/    # Asmamba / SA-SSM / 4D-8D-SSM 等
│   │   ├── decode_heads/ # UPerHeadDictEnhanced / UPerHeadDict
│   │   ├── losses/       # Dice / Focal / Lovasz / Tversky
│   │   └── utils/        # dynamic_dictionary(_enhanced) / adaptive_loss_weights
│   ├── datasets/         # custom_three_class(_oversample) + samplers/dcbs_*
│   ├── engine/hooks/     # dcbs_complement_hook / csv_logger_hook
│   ├── evaluation/  structures/  utils/  visualization/  registry/
├── tools/
│   ├── train.py          # 训练入口
│   └── test.py           # 测试 / 评估入口
├── data/                 # 数据占位目录（自行放入真实数据，见下文）
│   ├── img/{train,val,test}/
│   └── mask/{train,val,test}/
├── requirements/         # 依赖清单
├── requirements.txt
├── setup.py / setup.cfg / MANIFEST.in
├── run.sh                # 一键训练脚本
└── README.md
```

## 环境准备



```bash

conda activate Asmamba          # 
pip install -e .                # 可选：以可编辑方式安装 mmseg 包
```

依赖版本：

| 依赖 | 版本 |
|------|------|
| Python | 3.9 |
| PyTorch | 2.1.2+cu121 |
| mmcv | 2.1.0 |
| mmengine | 0.10.7 |
| timm | ≥0.9（含 DropPath/to_2tuple/register_model） |
| mamba_ssm | ≥1.0（提供 `Mamba` 算子） |
| einops | ≥0.6 |

> `Asmamba` backbone 依赖 `mamba_ssm`（CUDA 编译）与 `timm`，缺失会导致导入失败。
> 其余运行依赖见 [requirements/runtime.txt](requirements/runtime.txt) 与 [requirements/mminstall.txt](requirements/mminstall.txt)。

## 数据准备

配置文件使用 `data_root = 'data'`，需要把真实数据放入本工程 `data/` 目录：

```
data/
├── img/
│   ├── train/   *.png
│   ├── val/     *.png
│   └── test/    *.png
└── mask/
    ├── train/   *.png（与 img 同名）
    ├── val/     *.png
    └── test/    *.png
```

原始数据位于原仓库 `/home/zjl/envs/Samba-main/data/`（原仓库目录名未改动），可直接复制：

```bash
cp -r /home/zjl/envs/Samba-main/data/img/* data/img/
cp -r /home/zjl/envs/Samba-main/data/mask/* data/mask/
```

## 训练

```bash
# 一键脚本
bash run.sh

# 或手动执行
python tools/train.py configs/Asmamba_sa_ssm_upernet_dict_enhanced.py
```

- 默认单卡训练 80000 iter，输出目录 `output/Asmamba_sa_ssm_upernet_dict_enhanced/`
- 指定其它输出目录：`python tools/train.py <config> --work-dir <dir>`
- 断点续训：`python tools/train.py <config> --work-dir <dir> --resume`

## 测试 / 评估

```bash
python tools/test.py \
    configs/Asmamba_sa_ssm_upernet_dict_enhanced.py \
    output/Asmamba_sa_ssm_upernet_dict_enhanced/best_mIoU_iter_78000.pth
```

输出 mIoU / mDice / mFscore，并保存分割可视化结果。

## 说明

- `configs/Asmamba_sa_ssm_upernet_dict_enhanced.py` 为训练时实际使用的**自包含配置**（由 mmengine 完整 dump），不依赖 `_base_` 配置链。
- 训练权重（`.pth`，约 13GB）为输出产物，未包含在本工程内；需要时从原 `output/` 目录复制。
- 模型从零初始化训练（配置中 `load_from = None`），无需预训练权重。
