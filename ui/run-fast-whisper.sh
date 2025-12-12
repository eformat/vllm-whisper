#!/bin/bash

# on latest fedora42, latest rhaiis - we get error in torch device detection!
# Linux virt 6.17.10-200.fc42.x86_64 #1 SMP PREEMPT_DYNAMIC Mon Dec  1 18:04:51 UTC 2025 x86_64 GNU/Linux
# NVIDIA-SMI 590.44.01              Driver Version: 590.44.01      CUDA Version: 13.1
# registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.4

podman run --rm \
  --device nvidia.com/gpu=0 \
  --security-opt label=type:nvidia_container_t \
  --privileged --network=host --pid=host \
  -p 8000:8000 \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --env "HF_HUB_OFFLINE=0" \
  -v ~/tmp/whisper:/opt/app-root/src:z \
  --name=rhaiis \
  registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.2-1765379088 \
  --model distil-whisper/distil-large-v3 \
  --max-num-seqs 1 --gpu-memory-utilization 0.95 --max-model-len=448 --enforce-eager

