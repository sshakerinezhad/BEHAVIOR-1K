#!/bin/bash

EXP_NAME="ds_replay"
LOG_DIR="video_outputs/${EXP_NAME}"

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

mkdir -p "${LOG_DIR}"

# POLICY_ARGS="policy=local policy_config=pi05_b1k policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251029_024836/25000/"
POLICY_ARGS="policy=lookup"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=picking_up_trash \
    eval_on_train_instances=true \
    eval_instance_ids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] \
    max_steps=1000 \
    extra_notes="The goal here is to see if the policy can use its arms when given a very low-level prompt." \
    log_path="${LOG_DIR}/${EXP_NAME}_tor_test" \
    # prompt="\"Spin around and around in circles.\""

# XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=turning_on_radio \
#     eval_on_train_instances=true \
#     eval_instance_ids=[0,1,2,3,4,5] \
#     log_path="${LOG_DIR}/${EXP_NAME}_tor_train"

# XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=picking_up_trash \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0,1,2,3] \
#     log_path="${LOG_DIR}/${EXP_NAME}_picking_up_trash_test"

# XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=freeze_pies \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0,1,2,3] \
#     log_path="${LOG_DIR}/${EXP_NAME}_fp_test"
