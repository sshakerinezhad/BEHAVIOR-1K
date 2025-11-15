#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

export CUDA_VISIBLE_DEVICES=3;
export B1K_EVAL_TIME=true;
export OMNIGIBSON_DATA_PATH=/opt/BEHAVIOR-1K/datasets;

########################################################
#
# Will keep the default 10 diffusion steps but using receeding_temporal control mode
#
########################################################

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k_oversample/openpi_05_20251114_055221/69000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k_oversample/openpi_05_20251114_055221/69000/

EXP_NAME="openpi_05_20251114_055221_69k_steps"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k_22_TASKS_oversample policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k_oversample/openpi_05_20251114_055221/69000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=moving_boxes_to_storage \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3,4,5,6] \
    use_heavy_robot=true \
    max_steps=5400 \
    inf_time_proprio_dropout=0.0 \
    num_diffusion_steps=10 \
    control_mode=receeding_temporal \
    log_path="${LOG_DIR}/test_10_steps_receeding_temporal_fixed"

########################################################
#
# Trying control_mode = receeding_horizon for hella steps and on hella instances
# but also using proprio dropout of 0.25
#
########################################################

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k_oversample/openpi_05_20251114_055221/69000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k_oversample/openpi_05_20251114_055221/69000/

EXP_NAME="openpi_05_20251114_055221_69k_steps"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k_22_TASKS_oversample policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k_oversample/openpi_05_20251114_055221/69000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=moving_boxes_to_storage \
    eval_on_train_instances=false \
    use_heavy_robot=true \
    inf_time_proprio_dropout=0.25 \
    control_mode=receeding_horizon \
    log_path="${LOG_DIR}/test_receeding_horizon_29k_steps_025_pd"
