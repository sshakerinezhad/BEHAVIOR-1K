#!/bin/bash

    # max_steps=500

# log_path="pi0_skills_48000"

deactivate
eval "$(conda shell.bash hook)"
conda deactivate
conda activate behavior
time XLA_PYTHON_CLIENT_PREALLOCATE=false python OmniGibson/omnigibson/learning/eval.py \
    policy=local \
    task.name=turning_on_radio \
    log_path="pi05_8000" \
