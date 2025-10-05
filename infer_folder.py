import argparse
import cv2
import os
from ultralytics import YOLO
from glob import glob

def infer_folder(model_path, folder, fps=None):
    model = YOLO(model_path)
    images = sorted(glob(os.path.join(folder, "*.jpg")) + glob(os.path.join(folder, "*.png")))

    if fps:
        h, w, _ = cv2.imread(images[0]).shape
        out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
    else:
        out = None

    os.makedirs("outputs", exist_ok=True)

    for img in images:
        results = model.predict(img, conf=0.25, imgsz=1280)
        annotated = results[0].plot()
        cv2.imwrite(os.path.join("outputs", os.path.basename(img)), annotated)
        if out:
            out.write(annotated)

    if out:
        out.release()
        print("Saved video: output.mp4")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train/crack_yolov8n/weights/best.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()

    infer_folder(args.weights, args.source, fps=args.fps)
