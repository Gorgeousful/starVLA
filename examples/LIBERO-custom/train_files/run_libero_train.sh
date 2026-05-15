

# Single-node training: do not pin NCCL to cluster-specific NIC / IB devices.
# Let NCCL pick a local socket interface and keep IB disabled for local machines.
unset NCCL_SOCKET_IFNAME
unset NCCL_IB_HCA
export NCCL_IB_DISABLE=1

# used for check save when communication
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=10000  # timeout set to 1 hour (unit: seconds)
export NCCL_SOCKET_TIMEOUT_MS=360000
###########################################################################################
# === Please modify the following paths according to your environment ===
config_yaml=./examples/LIBERO-custom/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all_custom # libero_all_custom
run_root_dir=./playground/Checkpoints
run_id=0513_libero4in1_custom_qwen3ki
wandb_entity=luokang2192-irmv
wandb_project=starvla_libero
per_device_batch_size=4
gradient_accumulation_steps=8
is_debug=False
deepspeed_config_yaml=./starVLA/config/deepseeds/deepspeed_zero2.yaml
# === End of environment variable configuration ===
###########################################################################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp $0 ${output_dir}/ # mv this script to the output dir


# export WANDB_MODE=disabled
export CUDA_VISIBLE_DEVICES=4,6 # 4,6
num_processes=${NUM_PROCESSES:-$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')}
accelerate launch \
  --config_file ${deepspeed_config_yaml} \
  --num_processes ${num_processes} \
  --main_process_port 29501 \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --datasets.vla_data.data_root_dir ${libero_data_root}\
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${per_device_batch_size} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project ${wandb_project} \
  --wandb_entity ${wandb_entity} \
  --is_debug ${is_debug}
