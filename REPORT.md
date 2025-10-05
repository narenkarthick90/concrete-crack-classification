# 🧱 Concrete Crack Detection – Project Report

## 1. Problem Overview

Concrete cracks are a critical structural defect that can lead to unsafe buildings and costly maintenance if not detected early.  
The objective of this project is to **detect** cracks in concrete surfaces using a deep learning model trained on a labeled image dataset.

Unlike classification (which only predicts “crack / no crack”), this project performs **object detection**, identifying the *location* of cracks within each image.  
The solution is optimized for **accuracy, speed, and compactness**, making it deployable on edge devices such as drones or surveillance systems.

---

## 2. Dataset

- **Source:** Kaggle – *Concrete Crack Images for Classification*  
  (manually adapted for detection by labeling bounding boxes for cracked regions)
- **Classes:** `1` → `crack`
- **Structure:**
data/
├── raw/
│ ├── images/
│ │ ├── train/
│ │ ├── val/
│ │ └── test/
│ └── labels/
│ ├── train/
│ ├── val/
│ └── test/
└── crack_dataset.yaml

- **Split:**  
- Train: 70%  
- Validation: 15%  
- Test: 15%

---

## 3. Model Architecture

- **Base model:** YOLOv8-Small (`yolov8s.pt`)
- **Framework:** [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- **Key layers:**
- Backbone: CSPDarknet
- Neck: PANet
- Head: YOLO detection heads for bounding box regression and classification

---

## 4. Training Configuration

| Parameter | Value |
|------------|--------|
| Epochs | 40 (with early stopping, patience=5) |
| Image size | 640 × 640 |
| Batch size | 16 |
| Optimizer | SGD (default YOLOv8) |
| Learning rate | 0.01 (cosine schedule) |
| Device | CUDA-enabled GPU |
| Framework | Ultralytics YOLOv8 (PyTorch backend) |

---

## 5. Performance Metrics

| Metric | In-domain (test set) | Cross-domain subset |
|---------|----------------------|--------------------|
| **mAP@0.5** | 0.74  *(Pass/Stretch achieved)* | 0.57  |
| **Precision** | 0.89 | 0.83 |
| **Recall** | 0.86 | 0.78 |
| **F1-score** | 0.875 | 0.805 |

### Interpretation
- **Pass criteria:** mAP@0.5 ≥ 0.65 / 0.50  
- **Stretch goal:** mAP@0.5 ≥ 0.70 / 0.55  
 Achieved stretch threshold on both.

---

## 6. Inference and Edge Benchmark

| Metric | Result | Requirement | Status |
|---------|---------|--------------|------|
| **Average inference time (ONNX)** | 7.63 ms / image | ≤ 50 ms |
| **FPS (frames per second)** | 131 FPS | ≥ 20 FPS |
| **Memory usage** | 105 MB | ≤ 250 MB |
| **ONNX model size** | 14.2 MB | ≤ 25 MB |

**Environment:**  
- CUDA 13.0  
- ONNX Runtime (CPUExecutionProvider)  
- GPU: NVIDIA RTX 4050 (8 GB)

---

## 7. Files Generated

| File | Description |
|------|--------------|
| `results/crack_detector/weights/best.pt` | Trained YOLOv8 weights |
| `crack_detector.onnx` | Exported ONNX model for deployment |
| `benchmark_onnx.py` | Script to measure inference time, FPS, and memory |
| `train_yolo.py` | Training and validation pipeline |
| `infer_yolo.py` | Image/folder inference and result saving |
| `report.md` | Final documentation report |

---

## 8. Key Observations

- The YOLOv8-Small model achieved high accuracy while keeping ONNX file size small (<15 MB).  
- Average inference time of 7.6 ms (≈131 FPS) makes it suitable for **real-time crack inspection**.  
- Model met both “Pass” and “Stretch” targets specified in the task document.  
- Deployment-ready for mobile/edge environments after TensorRT conversion.

---

## 9. Future Work

- Integrate model into a **video stream pipeline** for live detection.  
- Fine-tune on additional datasets (e.g., bridge or pavement cracks).  
- Apply quantization (INT8) for further model size reduction (<10 MB).  
- Add confidence-based alerting system for on-site crack monitoring.

---

## 10. Conclusion

The developed YOLOv8-based detector successfully identifies cracks in concrete structures with high precision and real-time performance.  
It fully satisfies the accuracy (mAP ≥ 0.70) and efficiency (ONNX ≤ 25 MB, FPS ≥ 20) benchmarks from the original specification.  
This model is now ready for integration into practical edge-based inspection systems.
