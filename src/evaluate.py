"""
Evaluation script for trained Clash of Clans Gold Mine detection/segmentation model.
Measures Precision, Recall, mAP50, and mAP50-95 for boxes and masks on validation/test sets.
"""

import argparse
import os
import sys
from ultralytics import YOLO


def evaluate(
    model_path: str = "models/gold_mine_best.pt",
    data_yaml: str = "data.yaml",
    split: str = "test",
    imgsz: int = 640,
):
    """Evaluate YOLO detector/segmenter on test or validation split."""
    if not os.path.exists(model_path):
        candidates = [
            "runs/segment/coc_gold_mine_detector/weights/best.pt",
            "runs/detect/coc_gold_mine_detector/weights/best.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break
        else:
            print(f"[ERROR] Trained model weights not found at '{model_path}'.")
            print("Please run 'python src/train.py' first.")
            sys.exit(1)

    print("=" * 60)
    print("Evaluating Gold Mine Model")
    print(f"Model      : {os.path.abspath(model_path)}")
    print(f"Data config: {os.path.abspath(data_yaml)}")
    print(f"Split      : {split}")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results Summary:")
    if hasattr(metrics, "box") and metrics.box is not None:
        print("  [Bounding Boxes]")
        print(f"    Precision    : {metrics.box.mp:.4f}")
        print(f"    Recall       : {metrics.box.mr:.4f}")
        print(f"    mAP @ 0.50   : {metrics.box.map50:.4f}")
        print(f"    mAP @ 0.50:95: {metrics.box.map:.4f}")

    if hasattr(metrics, "seg") and metrics.seg is not None:
        print("  [Segmentation Masks]")
        print(f"    Precision    : {metrics.seg.mp:.4f}")
        print(f"    Recall       : {metrics.seg.mr:.4f}")
        print(f"    mAP @ 0.50   : {metrics.seg.map50:.4f}")
        print(f"    mAP @ 0.50:95: {metrics.seg.map:.4f}")
    print("=" * 60)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate CoC Gold Mine detector/segmenter")
    parser.add_argument("--model", default="models/gold_mine_best.pt", help="Path to trained .pt weights")
    parser.add_argument("--data", default="data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", default="test", choices=["val", "test", "train"], help="Split to evaluate")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
