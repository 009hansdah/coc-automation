"""
Drop point calculator for Clash of Clans automated deployments.
Calculates the closest point (minimum perpendicular distance) on the
boundary/deployment perimeter lines for each detected Gold Mine.

Supports both point-to-perimeter projection and exact convex building-to-perimeter
boundary distance minimization (using bounding box or contour vertices).
"""

import math
from typing import Sequence, Tuple, List, Dict, Any
import numpy as np

# Coordinates from auto_actions.py:
# Top-Left -> Left -> Bottom-Left -> Bottom-Right -> Right -> Top-Right -> Top-Left
DEFAULT_PERIMETER = [
    (859, 0),      # 1. Top-Left (press + hold start)
    (80, 521),     # 2. Left
    (859, 1079),   # 3. New Bottom-Left
    (1062, 1079),  # 4. New Bottom-Right (y=1079 screen bottom edge)
    (1788, 535),   # 5. Right
    (1062, 0),     # 6. Top-Right (release point)
    (859, 0),      # 7. Loop closure along top edge
]


def closest_point_on_segment(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[Tuple[float, float], float]:
    """Find the point Q on line segment AB closest to point P, and the distance |P - Q|.

    Formula:
        u = P - A, v = B - A
        t = (u . v) / |v|^2, clamped to [0.0, 1.0]
        Q = A + t * v
        dist = |P - Q|
    """
    px, py = p
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy

    if denom <= 1e-9:
        dist = math.hypot(px - ax, py - ay)
        return (float(ax), float(ay)), dist

    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))

    qx = ax + t * dx
    qy = ay + t * dy
    dist = math.hypot(px - qx, py - qy)
    return (qx, qy), dist


def building_to_segment_min_distance(
    building: Any,
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> Tuple[Tuple[float, float], float, Tuple[float, float]]:
    """Compute exact minimum Euclidean distance between a building (box or polygon) and segment AB.

    For disjoint convex sets, the global minimum distance occurs at either:
      1. A vertex of the building projected onto segment AB, OR
      2. An endpoint of segment AB (A or B) projected onto the building edges.

    Args:
        building: Either a bounding box (x1, y1, x2, y2) or polygon sequence [(x, y), ...]
        a: Segment start point
        b: Segment end point

    Returns:
        (best_drop_point_on_segment, min_distance, closest_point_on_building)
    """
    if (
        isinstance(building, (list, tuple, np.ndarray))
        and len(building) >= 4
        and not isinstance(building[0], (list, tuple, np.ndarray))
    ):
        # Bounding box: (x1, y1, x2, y2)
        x1, y1, x2, y2 = float(building[0]), float(building[1]), float(building[2]), float(building[3])
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        edges = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
    else:
        # Polygon: sequence of points
        corners = [(float(p[0]), float(p[1])) for p in building]
        edges = list(zip(corners, corners[1:] + [corners[0]]))

    best_dist = float("inf")
    best_drop = (0.0, 0.0)
    best_mine_pt = (0.0, 0.0)

    # 1. Project all building vertices onto perimeter segment AB
    for c in corners:
        q, d = closest_point_on_segment(c, a, b)
        if d < best_dist:
            best_dist = d
            best_drop = q
            best_mine_pt = c

    # 2. Project perimeter endpoints A and B onto building edges
    for ep in (a, b):
        for e1, e2 in edges:
            p, d = closest_point_on_segment(ep, e1, e2)
            if d < best_dist:
                best_dist = d
                best_drop = ep
                best_mine_pt = p

    return best_drop, best_dist, best_mine_pt


def find_closest_perimeter_point(
    target: Any,
    perimeter: Sequence[Tuple[float, float]] = DEFAULT_PERIMETER,
) -> Tuple[Tuple[float, float], float, int, Tuple[float, float]]:
    """Find the point on the perimeter lines with minimal Euclidean distance to the building.

    Supports:
      - Point (cx, cy): projects point onto segments.
      - Bounding box (x1, y1, x2, y2): computes exact minimum boundary distance.
      - Polygon [(x, y), ...]: computes exact minimum polygon boundary distance.

    Returns:
        (best_drop_point, min_distance, segment_index, nearest_mine_point)
    """
    # Defensive loop closure: if perimeter does not loop back to start, close it
    segments = []
    for i in range(len(perimeter) - 1):
        segments.append((perimeter[i], perimeter[i + 1], i))
    if len(perimeter) >= 3 and perimeter[0] != perimeter[-1]:
        segments.append((perimeter[-1], perimeter[0], len(perimeter) - 1))

    is_single_point = (
        isinstance(target, (list, tuple))
        and len(target) == 2
        and not isinstance(target[0], (list, tuple, np.ndarray))
    )

    best_point = (0.0, 0.0)
    min_dist = float("inf")
    best_segment = -1
    best_mine_pt = (0.0, 0.0)

    for seg_a, seg_b, idx in segments:
        if is_single_point:
            pt, dist = closest_point_on_segment(target, seg_a, seg_b)
            mine_pt = (float(target[0]), float(target[1]))
        else:
            pt, dist, mine_pt = building_to_segment_min_distance(target, seg_a, seg_b)

        if dist < min_dist:
            min_dist = dist
            best_point = pt
            best_segment = idx
            best_mine_pt = mine_pt

    return best_point, min_dist, best_segment, best_mine_pt


def calculate_drop_points(
    mine_centers: Sequence[Any] = None,
    perimeter: Sequence[Tuple[float, float]] = DEFAULT_PERIMETER,
    boxes: Sequence[Any] = None,
    centers: Sequence[Any] = None,
) -> List[Dict[str, Any]]:
    """Compute optimal deployment drop points for all detected Gold Mines.

    Supports both bounding boxes (for exact edge-to-perimeter minimum distance)
    and centroid points.

    Can be called as:
      - calculate_drop_points(centers, perimeter)
      - calculate_drop_points(boxes, perimeter)
      - calculate_drop_points(boxes=boxes, centers=centers, perimeter=perimeter)

    Returns a list of dicts:
      {
        "mine_center": (cx, cy),
        "nearest_mine_point": (mx, my),
        "drop_point": (qx, qy),
        "distance": dist,
        "segment_index": seg_idx,
      }
    """
    if mine_centers is None and centers is not None:
        mine_centers = centers

    targets = boxes if (boxes is not None and len(boxes) > 0) else mine_centers
    if targets is None or len(targets) == 0:
        return []

    results = []
    first_item = targets[0]
    is_box = len(first_item) >= 4 and not isinstance(first_item[0], (list, tuple, np.ndarray))

    for item in targets:
        if is_box:
            b_coords = (float(item[0]), float(item[1]), float(item[2]), float(item[3]))
            cx = (b_coords[0] + b_coords[2]) / 2.0
            cy = (b_coords[1] + b_coords[3]) / 2.0
            drop_pt, dist, seg_idx, mine_pt = find_closest_perimeter_point(b_coords, perimeter)
        else:
            cx, cy = float(item[0]), float(item[1])
            drop_pt, dist, seg_idx, mine_pt = find_closest_perimeter_point((cx, cy), perimeter)

        results.append({
            "mine_center": (cx, cy),
            "nearest_mine_point": mine_pt,
            "drop_point": drop_pt,
            "distance": dist,
            "segment_index": seg_idx,
        })
    return results


def _self_check():
    """Verify closest point calculations with known geometric cases."""
    # 1. Perpendicular drop inside horizontal segment
    p1 = (100.0, 50.0)
    pt1, d1 = closest_point_on_segment(p1, (0.0, 0.0), (200.0, 0.0))
    assert math.isclose(pt1[0], 100.0) and math.isclose(pt1[1], 0.0), f"Failed: {pt1}"
    assert math.isclose(d1, 50.0), f"Failed dist: {d1}"

    # 2. Clamped to endpoint
    p2 = (-20.0, 50.0)
    pt2, _ = closest_point_on_segment(p2, (0.0, 0.0), (200.0, 0.0))
    assert math.isclose(pt2[0], 0.0) and math.isclose(pt2[1], 0.0)

    # 3. Exact box to segment distance
    box = (100.0, 100.0, 160.0, 160.0)
    drop_b, dist_b, mine_b = building_to_segment_min_distance(box, (0.0, 0.0), (300.0, 0.0))
    assert math.isclose(dist_b, 100.0), f"Expected 100.0, got {dist_b}"
    assert math.isclose(drop_b[1], 0.0)
    assert math.isclose(mine_b[1], 100.0)

    # 4. Perimeter closest check with new 6-sided boundary
    # Mine near the bottom horizontal segment (seg 2: (859, 1079) -> (1062, 1079))
    mine_bottom = (960.0, 950.0)
    best_pt, dist, seg, mpt = find_closest_perimeter_point(mine_bottom, DEFAULT_PERIMETER)
    assert seg == 2, f"Expected segment 2 for bottom mine, got {seg}"
    assert math.isclose(best_pt[0], 960.0) and math.isclose(best_pt[1], 1079.0)
    assert math.isclose(dist, 129.0)

    # 5. Box-based drop calculation
    res = calculate_drop_points(boxes=[box])
    assert len(res) == 1
    assert "nearest_mine_point" in res[0] and "drop_point" in res[0]

    print("[PASS] drop_calculator geometry self-check passed.")


if __name__ == "__main__":
    _self_check()
