#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

export B1K_EVAL_TIME=true

# –––––– Scratchpad ––––––
# POLICY_ARGS="policy=lookup"
# POLICY_ARGS="policy=websocket"

# –––––– Model 1 ––––––
aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/ \
    /workspace/openpi/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/

EXP_NAME="pi05_single_task_mbts_256_ah"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_single_task policy_dir=/workspace/openpi/outputs/checkpoints/pi05_single_task/openpi_05_20251109_221546/37000/"

# XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=moving_boxes_to_storage \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0] \
#     use_heavy_robot=true \
#     max_steps=4000 \
#     inf_time_proprio_dropout=0.0 \
#     log_path="${LOG_DIR}/${EXP_NAME}_mbts_test_0.0_proprio_dropout"

# XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=moving_boxes_to_storage \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0] \
#     use_heavy_robot=true \
#     max_steps=4000 \
#     inf_time_proprio_dropout=0.2 \
#     log_path="${LOG_DIR}/${EXP_NAME}_mbts_test_0.2_proprio_dropout"

# XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=moving_boxes_to_storage \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0] \
#     use_heavy_robot=true \
#     max_steps=4000 \
#     inf_time_proprio_dropout=0.5 \
#     log_path="${LOG_DIR}/${EXP_NAME}_mbts_test_0.5_proprio_dropout"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=moving_boxes_to_storage \
    eval_on_train_instances=true \
    eval_instance_ids=[0,1,2,3] \
    use_heavy_robot=true \
    max_steps=7000 \
    n_ds_steps=2900 \
    inf_time_proprio_dropout=0.2 \
    log_path="${LOG_DIR}/${EXP_NAME}_mbts_train_0.2_proprio_dropout_again"

# # –––––– Model 2 ––––––
# aws s3 sync \
#     s3://behavior-challenge/outputs/checkpoints/pi0_b1k_2nd/openpi_0_20251109_202831/45000/ \
#     /workspace/openpi/outputs/checkpoints/pi0_b1k_2nd/openpi_0_20251109_202831/45000/

# EXP_NAME="pi0_zero_proprio_256_ah"
# LOG_DIR="video_outputs/${EXP_NAME}"

# mkdir -p "${LOG_DIR}"

# POLICY_ARGS="policy=local policy_config=pi0_b1k_2nd policy_dir=/workspace/openpi/outputs/checkpoints/pi0_b1k_2nd/openpi_0_20251109_202831/45000/"

# XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=picking_up_trash \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0,1,2] \
#     use_heavy_robot=true \
#     log_path="${LOG_DIR}/${EXP_NAME}_putrash_test"

# XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=picking_up_toys \
#     eval_on_train_instances=test \
#     eval_instance_ids=[0,1,2] \
#     use_heavy_robot=true \
#     log_path="${LOG_DIR}/${EXP_NAME}_putoys_test"
