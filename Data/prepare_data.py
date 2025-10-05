import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
CLASSES = ["Positive", "Negative"]

def make_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            os.makedirs(os.path.join(PROC_DIR, split, cls), exist_ok=True)

def prepare_dataset():
    all_images, all_labels = [], []
    for cls in CLASSES:
        imgs = glob(os.path.join(RAW_DIR, cls, "*.jpg"))
        all_images.extend(imgs)
        all_labels.extend([cls]*len(imgs))

    # Train/val/test split 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        all_images, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )

    splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    # Make dirs
    make_dirs()

    # Copy files
    for split, (imgs, labels) in splits.items():
        for img, cls in zip(imgs, labels):
            dst = os.path.join(PROC_DIR, split, cls, os.path.basename(img))
            shutil.copy(img, dst)

    print("Dataset prepared at", PROC_DIR)

if __name__ == "__main__":
    prepare_dataset()
