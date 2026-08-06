#!/bin/bash
# 一键训练脚本：Asmamba + SA-SSM + UPerNet + Enhanced Dynamic Dictionary
set -e

CONFIG=configs/Asmamba_sa_ssm_upernet_dict_enhanced.py
PYTHON="${PYTHON:-python}"

echo "======================================================================"
echo " 训练 Asmamba + SA-SSM + UPerNet + Enhanced Dynamic Dictionary"
echo " 配置: $CONFIG"
echo "======================================================================"

# 检查数据目录是否就绪
if [ -z "$(ls -A data/img/train 2>/dev/null)" ]; then
    echo "⚠️  data/img/train 为空，请先将真实训练数据放入 data/ 目录（见 README.md）"
    exit 1
fi

exec $PYTHON tools/train.py "$CONFIG"
