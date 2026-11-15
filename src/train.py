"""
Training script for Clash of Clans Gold Mine detection.
Supports both bounding boxes (YOLO detect) and segmentation masks (YOLO segment).
Automatically selects the correct base model and optimizes settings for Windows GPU training.
"""

import argparse
import os
import shutil
import sys
import torch
from ultralytics import YOLO


def detect_dataset_task(labels_dir: str = os.path.join("dataset", "labels", "train")) -> str:
    """Check label files to determine if dataset uses segmentation masks or boxes.

    Returns 'segment' if polygons (>5 tokens per line) are found, else 'detect'.
    """
    if os.path.exists(labels_dir):
        for fname in os.listdir(labels_dir):
            if fname.lower().endswith(".txt"):
                fpath = os.path.join(labels_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        tokens = line.strip().split()
                        if len(tokens) > 5:
                            return "segment"
    return "detect"


def check_dataset_ready(data_yaml_path: str = "data.yaml") -> bool:
    """Verify that dataset folders have training images before launching training."""
    if not os.path.exists(data_yaml_path):
        print(f"[ERROR] '{data_yaml_path}' not found.")
        return False

    train_img_dir = os.path.join("dataset", "images", "train")
    if not os.path.exists(train_img_dir):
        print(f"[ERROR] Directory '{train_img_dir}' does not exist.")
        return False

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    train_images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(image_exts)]
    if not train_images:
        print(f"[WARNING] No images found in '{train_img_dir}'.")
        print("Please place your annotated dataset into:")
        print("  dataset/images/train/ (images)")
        print("  dataset/labels/train/ (YOLO .txt labels)")
        return False

    val_img_dir = os.path.join("dataset", "images", "val")
    val_images = [f for f in os.listdir(val_img_dir) if f.lower().endswith(image_exts)] if os.path.exists(val_img_dir) else []
    print(f"[OK] Found {len(train_images)} training images and {len(val_images)} validation images.")
    return True


def train(
    data_yaml: str = "data.yaml",
    base_model: str | None = None,
    epochs: int = 60,
    batch_size: int = 4,
    imgsz: int = 800,
    workers: int = 2,
    project_name: str = "coc_gold_mine_detector",
    device: str | None = None,
):
    """Run YOLO training for Gold Mine detection or segmentation."""
    task = detect_dataset_task()
    if base_model is None:
        base_model = "yolo11n-seg.pt" if task == "segment" else "yolo11n.pt"

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Clash of Clans - Gold Mine Model Training")
    print(f"Dataset Task: {task.upper()} ({'Segmentation Masks' if task == 'segment' else 'Bounding Boxes'})")
    print(f"Data config : {os.path.abspath(data_yaml)}")
    print(f"Base model  : {base_model}")
    print(f"Device      : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'CPU'})")
    print(f"Epochs      : {epochs}")
    print(f"Batch size  : {batch_size}")
    print(f"Image size  : {imgsz}")
    print(f"Workers     : {workers}")
    print("=" * 60)

    model = YOLO(base_model)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        workers=workers,
        name=project_name,
        device=device,
        plots=True,
        verbose=True,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir is None and hasattr(model, "trainer") and model.trainer is not None:
        save_dir = model.trainer.save_dir

    if save_dir:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        if os.path.exists(best_pt):
            os.makedirs("models", exist_ok=True)
            target_pt = os.path.join("models", "gold_mine_best.pt")
            shutil.copy(best_pt, target_pt)
            print("=" * 60)
            print("[SUCCESS] Training complete!")
            print(f"[+] Best weights saved to: {os.path.abspath(target_pt)}")
            print("=" * 60)
            return target_pt

    print("[SUCCESS] Training finished.")
    return None


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for CoC Gold Mine detection/segmentation")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default=None, help="Base model weights (default: yolo11n.pt or yolo11n-seg.pt)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4 for 4GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=800, help="Input image size (default: 800)")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers (default: 2 for Windows)")
    parser.add_argument("--device", default=None, help="Device to use: '0', 'cpu', etc.")
    parser.add_argument("--skip-check", action="store_true", help="Skip dataset readiness check")
    args = parser.parse_args()

    if not args.skip_check and not check_dataset_ready(args.data):
        sys.exit(1)

    train(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
