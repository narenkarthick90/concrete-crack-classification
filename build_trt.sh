#!/bin/bash
ONNX_MODEL="crack_classifier.onnx"
ENGINE="crack_classifier.plan"

trtexec --onnx=$ONNX_MODEL \
        --saveEngine=$ENGINE \
        --fp16 \
        --shapes=input:1x3x224x224 \
        --workspace=1024 > benchmark.log

echo "TensorRT engine built: $ENGINE"
