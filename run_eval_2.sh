#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

# Trying k = 0.05 again but with temporal ensemble here
export VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json;
export CUDA_VISIBLE_DEVICES=1;
export B1K_EVAL_TIME=true;
# export OMNIGIBSON_DATA_PATH=/opt/BEHAVIOR-1K/datasets;

export TRAIN_CONFIG_NAME="pi05_b1k_oversample_tor";
export CKPT_NAME="openpi_05_20251115_050323";
export STEP_COUNT=9000;
export TASK_NAME="turning_on_radio";

# export CONTROL_MODE="receeding_temporal";
# export MAX_LEN=100;
# export ACTION_HORIZON=20;
# export TEMPORAL_ENSEMBLE_MAX=5;
# export EXP_K_VALUE=0.2;

export CONTROL_MODE="receeding_horizon";
export MAX_LEN=50;
export ACTION_HORIZON=50;
export TEMPORAL_ENSEMBLE_MAX=1;
export EXP_K_VALUE=1.0;

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}/ \
    /workspace/openpi/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}/

export EXP_NAME="${TASK_NAME}";
export LOG_DIR="minimal_rollout_from_b1k/${EXP_NAME}";

mkdir -p "${LOG_DIR}";

export POLICY_ARGS="policy=local policy_config=pi05_b1k_inference_final policy_dir=/workspace/openpi/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}";

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name="${TASK_NAME}" \
    eval_on_train_instances=true \
    eval_instance_ids=[0] \
    use_heavy_robot=true \
    inf_time_proprio_dropout=0.0 \
    num_diffusion_steps=5 \
    control_mode=${CONTROL_MODE} \
    action_horizon=${ACTION_HORIZON} \
    max_len=${MAX_LEN} \
    temporal_ensemble_max=${TEMPORAL_ENSEMBLE_MAX} \
    exp_k_value=${EXP_K_VALUE} \
    log_path="${LOG_DIR}/second_jax_fix_v1"
