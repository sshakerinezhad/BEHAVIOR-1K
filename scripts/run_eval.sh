#!/bin/bash

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

# –––––– Scratchpad ––––––
# POLICY_ARGS="policy=lookup"
# POLICY_ARGS="policy=websocket"

# –––––– Model 1 ––––––
EXP_NAME="good_rl_40_step"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k policy_dir=/workspace/openpi/logs/20251108_022813/test_openpi_pi05_behavior/checkpoints/global_step_40/actor/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=picking_up_trash \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2] \
    log_path="${LOG_DIR}/${EXP_NAME}_put_test"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=turning_on_radio \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3,4] \
    log_path="${LOG_DIR}/${EXP_NAME}_tor_test"

# XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=freeze_pies \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0,1,2,3] \
#     log_path="${LOG_DIR}/${EXP_NAME}_fp_test"

# –––––– Model 2 ––––––
EXP_NAME="simple_run_arms_ds_filtering_no_weighting"
LOG_DIR="video_outputs/${EXP_NAME}"

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251108_052323/16000/"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=picking_up_trash \
    eval_on_train_instances=true \
    eval_instance_ids=[0,1,2] \
    log_path="${LOG_DIR}/${EXP_NAME}_put_test" \
    use_heavy_robot=true

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=turning_on_radio \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3,4] \
    use_heavy_robot=true \
    log_path="${LOG_DIR}/${EXP_NAME}_tor_test"

XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=freeze_pies \
    eval_on_train_instances=false \
    eval_instance_ids=[0,1,2,3] \
    use_heavy_robot=true \
    log_path="${LOG_DIR}/${EXP_NAME}_fp_test"
