# VARH-AI: NTIRE 2026 Efficient SR Challenge Submission

This repository contains the official evaluation code and pre-trained weights for Team **VARH-AI** (Team ID: 15).

## Methodology Summary
Our solution builds upon the architectural foundation of Team EMSR (the winner of the 2025 ESR challenge). We successfully introduced several primary optimizations to drive down inference latency and parameter size for the 2026 challenge:

1. **Exact Mathematical Operator Fusion**: We isolated sequential structural patterns in the architecture (such as linearly dependent 3x3 convolutions) and mathematically fused them into singular higher-order equivalent kernels (like 5x5s). This allowed us to explicitly drop thousands of parameters natively without altering the tensor output distribution.
2. **ConvLoRA & MuAdamW Distillation**: We utilized a ConvLORA-based student-teacher distillation pipeline, uniquely updating gradients using the recent hybrid `MuAdamW` optimizer (inspired by Ultralytics YOLO26).

   * **Performance Impact**: Evaluated natively on a single **NVIDIA L40S** GPU, operating in this optimized FP32 configuration yields a latency of **3.86 ms** per image and a validation-set PSNR of **26.92 dB**. Parameters: 125,992. FLOPs: 8.22 G.

## Evaluation Instructions

### 1. Requirements
Our solution strictly adheres to the PyTorch 1.13 + cu117 evaluation environment stipulated by the challenge organizers. 

### 2. Running Inference
To reproduce our predictions, please use the provided `team15_test_demo.py` script. The script is configured to automatically load our `team15_DSCF_Fused.pth` checkpoint and natively execute the fast FP32 evaluations.

```bash
python team15_test_demo.py \
    --data_dir /path/to/competition_data \
    --save_dir ./results \
    --model_id 15
```

### 3. Output
The script processes the validation test images, prints the inference timings, memory benchmarks, and PSNR calculations directly to the console (`results.json` & `results.txt`), entirely fulfilling the NTIRE guidelines.
