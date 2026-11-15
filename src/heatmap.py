"""
Heatmap generation module for Clash of Clans building detection.
Renders Gaussian heat spots centered on detected buildings.
"""

import cv2
import numpy as np


def generate_heatmap(
    frame: np.ndarray,
    centers: list[tuple[float, float]],
    radius: int = 45,
    sigma: int = 20,
    blend_only_hot: bool = True,
    alpha: float = 0.55,
) -> np.ndarray:
    """Generate and blend a Gaussian location heatmap for detected building centers.

    Args:
        frame: Original BGR image/frame (uint8).
        centers: List of (x, y) building center coordinates.
        radius: Base circle radius for each detected building.
        sigma: Gaussian blur standard deviation.
        blend_only_hot: If True, only blends where heat is present, keeping the
            rest of the base clear. If False, applies standard full-frame blend.
        alpha: Maximum opacity of the heatmap overlay (0.0 to 1.0).

    Returns:
        Annotated BGR frame with heatmap overlay.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Invalid input frame.")

    if not centers:
        return frame.copy()

    height, width = frame.shape[:2]
    heat = np.zeros((height, width), dtype=np.float32)

    for cx, cy in centers:
        x, y = int(round(cx)), int(round(cy))
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(heat, (x, y), radius, 1.0, -1)

    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma)
    max_val = heat.max()
    if max_val <= 0:
        return frame.copy()

    # Normalize to 0-255 uint8
    heat_norm = (heat / max_val * 255.0).astype(np.uint8)
    colored_heat = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

    if blend_only_hot:
        # Intensity-based alpha blending: hot spots glow, unheated areas remain untouched
        heat_factor = (heat_norm.astype(np.float32) / 255.0)[..., np.newaxis]
        blend_alpha = heat_factor * alpha
        out = frame.astype(np.float32) * (1.0 - blend_alpha) + colored_heat.astype(np.float32) * blend_alpha
        return np.clip(out, 0, 255).astype(np.uint8)
    else:
        # Standard full-screen blend from guide
        return cv2.addWeighted(frame, 1.0 - alpha, colored_heat, alpha, 0)

