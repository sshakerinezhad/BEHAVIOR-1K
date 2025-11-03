#!/bin/bash

EXP_NAME="25k_ds_filtering_again_special_prompts"
LOG_DIR="video_outputs/${EXP_NAME}"

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

mkdir -p "${LOG_DIR}"

POLICY_ARGS="policy=local policy_config=pi05_b1k policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251029_024836/25000/"

python OmniGibson/omnigibson/learning/eval.py \
    ${POLICY_ARGS} \
    task.name=turning_on_radio \
    eval_on_train_instances=false \
    eval_instance_ids=[0] \
    max_steps=1000 \
    extra_notes="The goal here is to see if the policy can use its arms when given a very low-level prompt." \
    log_path="${LOG_DIR}/${EXP_NAME}_tor_test" \
    prompt="\"Lift up and raise the radio receiver that's on the table in the living room.\""

# python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=turning_on_radio \
#     eval_on_train_instances=true \
#     eval_instance_ids=[0,1,2,3,4,5] \
#     log_path="${LOG_DIR}/${EXP_NAME}_tor_train"

# python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=picking_up_trash \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0,1,2,3] \
#     log_path="${LOG_DIR}/${EXP_NAME}_picking_up_trash_test"

# python OmniGibson/omnigibson/learning/eval.py \
#     ${POLICY_ARGS} \
#     task.name=freeze_pies \
#     eval_on_train_instances=false \
#     eval_instance_ids=[0,1,2,3] \
#     log_path="${LOG_DIR}/${EXP_NAME}_fp_test"
