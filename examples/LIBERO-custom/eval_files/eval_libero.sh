#!/bin/bash
STARVLA_DIR=/data0/luokang/research/starvla/starVLA
cd ${STARVLA_DIR}
# CKPT=${STARVLA_DIR}/playground/Checkpoints/0405_libero4in1_CosmoPredict2GR00T/checkpoints/steps_50000_pytorch_model.pt
CKPT=${STARVLA_DIR}/playground/Checkpoints/0513_libero4in1_custom_qwen3ki/checkpoints/steps_20000_pytorch_model.pt
###########################################################################################
export LIBERO_HOME=/data0/luokang/research/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=/data0/luokang/miniconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

host="127.0.0.1"
base_port=6694
unnorm_key="franka"
your_ckpt=${CKPT}

# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}') # model_root: playground/Checkpoints/<run_id>
###########################################################################################

task_suite_name=libero_object
num_trials_per_task=50
video_out_path="${model_root}/results/${task_suite_name}/${folder_name}"
mkdir -p "$video_out_path"
find "$video_out_path" -mindepth 1 -delete
log_file="${video_out_path}/eval_$(date +%Y%m%d_%H%M%S).log"

${LIBERO_Python} ./examples/LIBERO-custom/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path" \
    2>&1 | tee "$log_file"
