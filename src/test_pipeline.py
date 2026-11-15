"""
Self-check pipeline test for Clash of Clans Gold Mine detection system.
Runs assert-based verification on YAML configs, directory layout,
heatmap generation logic, drop point calculations, and YOLO model inference.
"""

import os
import sys
import numpy as np

# Ensure src is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from heatmap import generate_heatmap
from drop_calculator import DEFAULT_PERIMETER, calculate_drop_points, closest_point_on_segment
from ultralytics import YOLO
from ultralytics.data.utils import YAML


def test_data_yaml():
    yaml_path = "data.yaml"
    assert os.path.exists(yaml_path), f"Missing {yaml_path}"
    data = YAML.load(yaml_path)
    assert "names" in data, "data.yaml must contain 'names'"
    assert 0 in data["names"] or "0" in data["names"], "data.yaml must define class 0"
    cls_name = data["names"].get(0) or data["names"].get("0")
    assert cls_name == "gold_mine", f"Class 0 should be 'gold_mine', got {cls_name}"
    print("[PASS] data.yaml configuration verified.")


def test_directory_structure():
    required_dirs = [
        os.path.join("dataset", "raw"),
        os.path.join("dataset", "images", "train"),
        os.path.join("dataset", "images", "val"),
        os.path.join("dataset", "images", "test"),
        os.path.join("dataset", "labels", "train"),
        os.path.join("dataset", "labels", "val"),
        os.path.join("dataset", "labels", "test"),
        "models",
        "src",
    ]
    for d in required_dirs:
        assert os.path.isdir(d), f"Missing required directory: {d}"
    print("[PASS] Dataset directory structure verified.")


def test_heatmap_generation():
    h, w = 480, 640
    frame = np.full((h, w, 3), 120, dtype=np.uint8)

    # 1. Empty centers check
    out_empty = generate_heatmap(frame, [])
    assert out_empty.shape == frame.shape, "Empty centers shape mismatch"
    assert out_empty.dtype == np.uint8, "Empty centers dtype mismatch"
    assert np.array_equal(out_empty, frame), "Empty centers should return unmodified frame copy"

    # 2. Populated centers check
    centers = [(100.0, 150.0), (320.0, 240.0)]
    out_hot = generate_heatmap(frame, centers, radius=30, sigma=15, blend_only_hot=True)
    assert out_hot.shape == frame.shape
    assert out_hot.dtype == np.uint8
    assert not np.array_equal(out_hot, frame), "Heatmap should modify frame at centers"

    # 3. Out of bounds centers should not crash
    out_oob = generate_heatmap(frame, [(-50, -50), (9999, 9999)])
    assert out_oob.shape == frame.shape

    print("[PASS] Heatmap generator tests passed.")


def test_drop_point_calculations():
    # Test perpendicular projection
    pt, dist = closest_point_on_segment((100.0, 50.0), (0.0, 0.0), (200.0, 0.0))
    assert pt == (100.0, 0.0), f"Expected (100, 0), got {pt}"
    assert dist == 50.0

    # Test full perimeter calculation with mock Gold Mines
    mines = [(500.0, 200.0), (1200.0, 800.0)]
    results = calculate_drop_points(mines, DEFAULT_PERIMETER)
    assert len(results) == 2
    for r in results:
        assert "drop_point" in r and "distance" in r
        assert r["distance"] > 0
    print("[PASS] Drop point geometry calculations verified.")


def test_yolo_inference_dummy():
    weights = "yolo11n.pt"
    assert os.path.exists(weights), f"Weights file {weights} not found"
    model = YOLO(weights)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(dummy_img, imgsz=640, verbose=False)
    assert len(results) == 1, "Predict should return one result for single frame"
    assert hasattr(results[0], "boxes"), "Result should have 'boxes' attribute"
    print("[PASS] YOLO inference on synthetic image passed.")


def main():
    print("=" * 60)
    print("Running CoC Pipeline Self-Checks...")
    print("=" * 60)
    test_data_yaml()
    test_directory_structure()
    test_heatmap_generation()
    test_drop_point_calculations()
    test_yolo_inference_dummy()
    print("=" * 60)
    print("[ALL CHECKS PASSED] System is ready and verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
