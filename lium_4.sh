#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

export CUDA_VISIBLE_DEVICES=3;
export B1K_EVAL_TIME=true;
# export OMNIGIBSON_DATA_PATH=/opt/BEHAVIOR-1K/datasets;

export TRAIN_CONFIG_NAME="pi05_b1k_oversample_ltc";
export CKPT_NAME="ltc_openpi_05_20251116_073405";
export STEP_COUNT=15000;
export TASK_NAME="loading_the_car";

# export CONTROL_MODE="receeding_horizon";
# export MAX_LEN=100;
# export ACTION_HORIZON=100;
# export TEMPORAL_ENSEMBLE_MAX=1;
# export EXP_K_VALUE=1.0;

export CONTROL_MODE="receeding_temporal";
export MAX_LEN=72;
export ACTION_HORIZON=12;
export TEMPORAL_ENSEMBLE_MAX=6;
export EXP_K_VALUE=0.5;

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}/ \
    /workspace/openpi/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}/

export EXP_NAME="${TASK_NAME}_FINAL";
export LOG_DIR="final_video_outputs/${EXP_NAME}";

mkdir -p "${LOG_DIR}";

# export POLICY_ARGS="policy=local policy_config=pi05_b1k_inference_final policy_dir=/workspace/openpi/outputs/checkpoints/${TRAIN_CONFIG_NAME}/${CKPT_NAME}/${STEP_COUNT}";
export POLICY_ARGS="policy=websocket websockets_host=0.0.0.0 websockets_port=8003";

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name="${TASK_NAME}" \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3,4,5,6,7,8,9] \
    use_heavy_robot=true \
    inf_time_proprio_dropout=0.0 \
    num_diffusion_steps=10 \
    control_mode=${CONTROL_MODE} \
    action_horizon=${ACTION_HORIZON} \
    max_len=${MAX_LEN} \
    temporal_ensemble_max=${TEMPORAL_ENSEMBLE_MAX} \
    exp_k_value=${EXP_K_VALUE} \
    log_path="${LOG_DIR}/k_0.5"
