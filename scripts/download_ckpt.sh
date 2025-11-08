#!/bin/bash

# aws s3 sync \
#     s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20251001_035802/49999/ \
#     /workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251001_035802/49999/

aws s3 sync \
    s3://behavior-challenge/logs/20251108_022813/test_openpi_pi05_behavior/checkpoints/global_step_40/actor/ \
    /workspace/openpi/logs/20251108_022813/test_openpi_pi05_behavior/checkpoints/global_step_40/actor/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20251108_052323/16000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251108_052323/16000/
