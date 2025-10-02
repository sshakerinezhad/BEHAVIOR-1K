#!/bin/bash

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20251001_035802/49999/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251001_035802/49999/

aws s3 sync \
    s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856//35000/ \
    /workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20250929_205856/35000/

# aws s3 sync \
#     s3://behavior-challenge/outputs/checkpoints/pi05_b1k/openpi_05_20251001_035802/4000/ \
#     /workspace/openpi/outputs/checkpoints/pi05_b1k/openpi_05_20251001_035802/4000/
