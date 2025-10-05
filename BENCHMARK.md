# ONNX Benchmark Results

This report summarizes the performance benchmark of an ONNX model, as executed by the `benchmark_onnx.py` script.

---

## Performance Summary

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Average Inference Time** | **7.63 ms** | The time taken to process a single input. |
| **Frames Per Second (FPS)** | **131.11** | The number of inferences the model can perform per second. |
| **Memory Usage** | **105.33 MB** | The memory allocated by the ONNX Runtime session. |

---

## Execution Environment Warning 

The execution logged a **UserWarning** indicating a mismatch between the desired hardware acceleration and the available providers.

**Warning Message:**
