#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export STARVLA_DATA_REGISTRY_BENCH=LIBERO-custom
# === Paths (adapted for this cluster) ===
STARVLA_DIR=/data0/luokang/research/starvla/starVLA
LIBERO_HOME=/data0/luokang/research/LIBERO
STARVLA_PYTHON=/data0/luokang/miniconda3/envs/starvla/bin/python
LIBERO_PYTHON=/data0/luokang/micromamba/envs/libero/bin/python

# === Checkpoint ===
# CKPT=${STARVLA_DIR}/playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
CKPT=${STARVLA_DIR}/playground/Checkpoints/0513_libero4in1_custom_qwen3ki/checkpoints/steps_100000_pytorch_model.pt

export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}   
gpu_id=6
port=6694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
