"""
Inference script to run Gold Mine detection, segmentation masks, heatmap,
and optimal troop drop points on static images.
"""

import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from heatmap import generate_heatmap
    from drop_calculator import DEFAULT_PERIMETER, calculate_drop_points
except ImportError:
    from src.heatmap import generate_heatmap
    from src.drop_calculator import DEFAULT_PERIMETER, calculate_drop_points


def detect_on_image(
    image_path: str,
    model_path: str = "models/gold_mine_best.pt",
    conf: float = 0.45,
    imgsz: int = 800,
    output_dir: str = "dataset/detections",
    show: bool = False,
):
    """Run Gold Mine detector on an image and draw boxes, heatmap, and closest drop points."""
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: {image_path}")
        return

    if not os.path.exists(model_path):
        candidates = [
            "runs/detect/coc_gold_mine_detector/weights/best.pt",
            "runs/detect/coc_gold_mine_detector-2/weights/best.pt",
            "runs/segment/coc_gold_mine_detector/weights/best.pt",
            "models/best.pt",
            "yolo11n.pt",
            "yolo11n-seg.pt",
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    print(f"Processing '{image_path}' with model '{model_path}'...")
    model = YOLO(model_path)
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Could not decode image: {image_path}")
        return

    results = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)
    result = results[0]

    centers = []
    boxes = []
    polygons = []
    class_names = getattr(model, "names", {0: "gold_mine"})

    has_masks = result.masks is not None and len(result.masks) > 0
    has_boxes = result.boxes is not None and len(result.boxes) > 0

    if has_boxes:
        for idx, box in enumerate(result.boxes):
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            if cls_id == 0 or class_names.get(cls_id) == "gold_mine":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append((int(x1), int(y1), int(x2), int(y2), score))
                centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

                if has_masks and idx < len(result.masks.xy):
                    poly = result.masks.xy[idx]
                    if len(poly) > 0:
                        polygons.append(np.int32(poly))

    # 1. Apply Heatmap
    annotated = generate_heatmap(frame, centers, radius=45, sigma=20, blend_only_hot=True)

    # 2. Draw Imaginary Perimeter Line (Top -> Left -> Bottom -> Right -> Top)
    for i in range(len(DEFAULT_PERIMETER) - 1):
        p1 = (int(DEFAULT_PERIMETER[i][0]), int(DEFAULT_PERIMETER[i][1]))
        p2 = (int(DEFAULT_PERIMETER[i + 1][0]), int(DEFAULT_PERIMETER[i + 1][1]))
        cv2.line(annotated, p1, p2, (255, 230, 0), 2)

    # 3. Compute and Draw Drop Points & Guidelines (box-to-perimeter exact minimum)
    drop_data = calculate_drop_points(boxes=boxes, centers=centers, perimeter=DEFAULT_PERIMETER)
    for d in drop_data:
        m_pt = d.get("nearest_mine_point", d["mine_center"])
        mx, my = int(m_pt[0]), int(m_pt[1])
        qx, qy = int(d["drop_point"][0]), int(d["drop_point"][1])
        # Perpendicular guideline
        cv2.line(annotated, (mx, my), (qx, qy), (0, 255, 100), 1)
        # Highlight drop point dot
        cv2.circle(annotated, (qx, qy), 7, (0, 255, 100), -1)
        cv2.circle(annotated, (qx, qy), 9, (255, 255, 255), 2)

    # 4. Draw Masks
    if polygons:
        mask_overlay = annotated.copy()
        for poly in polygons:
            cv2.fillPoly(mask_overlay, [poly], (0, 215, 255))
            cv2.polylines(annotated, [poly], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.addWeighted(mask_overlay, 0.35, annotated, 0.65, 0, dst=annotated)

    # 5. Draw Boxes
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 2)
        label = f"Gold Mine {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w + 6, y1), (0, 215, 255), -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    out_filename = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
    cv2.imwrite(out_filename, annotated)
    print(f"[+] Detected {len(boxes)} Gold Mines with {len(drop_data)} drop points. Saved to: {out_filename}")

    if show:
        cv2.imshow("Detected Gold Mines & Drop Points", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Run detection and drop point calculation on a static image")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="models/gold_mine_best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference resolution")
    parser.add_argument("--output", default="dataset/detections", help="Output directory")
    parser.add_argument("--show", action="store_true", help="Display output window")
    args = parser.parse_args()

    detect_on_image(
        args.image,
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        output_dir=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
