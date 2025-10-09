#!/bin/bash

    #     max_steps=500 \
    #     eval_on_train_instances=true \

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior

# time XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     policy=local \
#     task.name=turning_on_radio \
#     eval_on_train_instances=true \
#     log_notes="tor_obv_training" \
#     policy_config=pi0_b1k \
#     policy_dir=/workspace/openpi/outputs/checkpoints/pi0_b1k/openpi_0_20251005_045853/46000/

# time XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     policy=local \
#     task.name=turning_on_radio \
#     env_wrapper._target_=omnigibson.learning.wrappers.DefaultWrapper \
#     eval_on_train_instances=true \
#     log_notes="training_with_default_wrapper" \
#     policy_config=pi05_b1k \
#     max_steps=2000 \
#     policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/49999/

time XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    policy=local \
    task.name=turning_on_radio \
    eval_on_train_instances=true \
    log_notes="training_with_dataset_inputs_proprio_only" \
    policy_config=pi05_b1k \
    max_steps=2000 \
    use_dataset_inputs_proprio_only=false \
    policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/49999/

# time XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     policy=local \
#     task.name=turning_on_radio \
#     log_notes="tor_subtask_prompts" \
#     policy_config=pi05_b1k \
#     policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251001_035802/77000/

# time XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     policy=local \
#     task.name=turning_on_radio \
#     log_notes="tor_subtask_prompts" \
#     policy_config=pi05_b1k \
#     policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/48000/

# time XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     policy=local \
#     task.name=turning_on_radio \
#     log_notes="tor" \
#     policy_config=pi05_b1k \
#     policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251001_035802/79000/

# time XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
#     policy=local \
#     task.name=freeze_pies \
#     log_notes="freeze_pies" \
#     policy_config=pi05_b1k \
#     policy_dir=/workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251001_035802/49999/
