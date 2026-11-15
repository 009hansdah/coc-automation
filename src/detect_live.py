"""
Real-time Clash of Clans Gold Mine detection and live HUD overlay.
Renders detection bounding boxes, labels, heat indicators, and optimal
troop deployment drop points DIRECTLY ON TOP of the game application.

Features:
  - Transparent Click-Through Overlay:
      * Hovers directly over the live game screen.
      * Mouse clicks and game controls pass through to Clash of Clans with zero interference.
      * Invisible to screen capture (no recursive screen mirroring).
  - Optimal Troop Drop Points (from auto_actions.py perimeter):
      * Computes the point on the deployment perimeter with the shortest perpendicular
        distance to each detected Gold Mine.
      * Displays the imaginary deployment boundary line.
      * Highlights each optimal drop position with a bright neon indicator dot.
      * Draws trajectory guide lines connecting each mine to its closest drop point.
  - Interactive Controls (Global - works while playing):
      * [H]          : Toggle Heatmap on/off
      * [B]          : Toggle Bounding Boxes on/off
      * [M]          : Toggle Segmentation Masks on/off
      * [D]          : Toggle Drop Points & Perimeter Line on/off
      * [+] / [-]    : Increase / Decrease confidence threshold
      * [F8] or [S]  : Save detection snapshot
      * [Q] or [Esc] : Quit overlay
"""

import argparse
import ctypes
import os
import queue
import sys
import threading
import time
import tkinter as tk
import cv2
import mss
import numpy as np
import torch
from ultralytics import YOLO

# Audio feedback on Windows
try:
    import winsound

    def play_beep():
        try:
            winsound.Beep(1200, 70)
        except Exception:
            pass
except ImportError:
    def play_beep():
        pass

# Global keyboard listener
try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False

# Import local heatmap generator & drop calculator
try:
    from heatmap import generate_heatmap
    from drop_calculator import DEFAULT_PERIMETER, calculate_drop_points
except ImportError:
    from src.heatmap import generate_heatmap
    from src.drop_calculator import DEFAULT_PERIMETER, calculate_drop_points


def get_default_save_dir() -> str:
    """Return absolute path to dataset/detections/live."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, "dataset", "detections", "live")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def find_model_weights(preferred_path: str = "models/gold_mine_best.pt") -> str:
    """Find available trained model weights or fallback to base model."""
    candidates = [
        preferred_path,
        "runs/detect/coc_gold_mine_detector/weights/best.pt",
        "runs/detect/coc_gold_mine_detector-2/weights/best.pt",
        "runs/segment/coc_gold_mine_detector/weights/best.pt",
        "models/best.pt",
        "yolo11n.pt",
        "yolo11n-seg.pt",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return preferred_path


# ==============================================================================
# Transparent Game Overlay Engine (Tkinter + Windows Win32 API)
# ==============================================================================

def setup_transparent_window(root: tk.Tk, left: int, top: int, width: int, height: int):
    """Configure a Tkinter window as a transparent, topmost, click-through overlay."""
    root.overrideredirect(True)
    root.geometry(f"{width}x{height}+{left}+{top}")
    root.wm_attributes("-topmost", True)
    root.wm_attributes("-transparentcolor", "black")
    root.config(bg="black")
    root.update()

    user32 = ctypes.windll.user32
    hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()

    # Click-through: WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOPMOST
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOPMOST = 0x00000008

    current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(
        hwnd,
        GWL_EXSTYLE,
        current_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST,
    )

    # Exclude overlay from screen capture so MSS grabs clean game frames (no mirror loops)
    try:
        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    except Exception:
        pass


def run_live_overlay(
    model_path: str,
    conf_threshold: float = 0.45,
    imgsz: int = 800,
    monitor_idx: int = 1,
    device: str | None = None,
    save_dir: str | None = None,
):
    """Run real-time transparent game overlay detection with closest drop points."""
    if save_dir is None:
        save_dir = get_default_save_dir()
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("   Clash of Clans - Live Game Transparent Overlay & Drop Points")
    print("=" * 65)
    print(f"Model weights   : {os.path.abspath(model_path)}")
    print(f"Device          : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != 'cpu' else 'CPU'})")
    print(f"Initial Conf    : {conf_threshold:.2f}")
    print(f"Perimeter Lines : {len(DEFAULT_PERIMETER)-1} segments (from auto_actions.py)")
    print("-" * 65)
    print("CONTROLS (Works while playing Clash of Clans):")
    print("  [D]          : Toggle Drop Points & Perimeter on/off")
    print("  [H]          : Toggle Heatmap on/off")
    print("  [B]          : Toggle Bounding Boxes on/off")
    print("  [M]          : Toggle Segmentation Masks on/off")
    print("  [+] / [-]    : Adjust confidence threshold")
    print("  [F8] or [S]  : Save detection snapshot image")
    print("  [Q] or [Esc] : Quit overlay")
    print("=" * 65)

    model = YOLO(model_path)
    class_names = getattr(model, "names", {0: "gold_mine"})
    target_class_id = 0

    results_queue = queue.Queue(maxsize=1)
    snapshot_trigger = threading.Event()
    quit_event = threading.Event()

    state = {
        "conf": conf_threshold,
        "show_heatmap": True,
        "show_boxes": True,
        "show_masks": True,
        "show_drop_points": True,
        "last_snapshot_msg": "",
        "last_snapshot_time": 0.0,
        "snapshot_count": 0,
        "fps": 0.0,
    }

    # Setup monitor geometry
    with mss.mss() as sct_init:
        if monitor_idx >= len(sct_init.monitors):
            monitor_idx = 1
        mon_info = sct_init.monitors[monitor_idx]
        mon_left = mon_info["left"]
        mon_top = mon_info["top"]
        mon_width = mon_info["width"]
        mon_height = mon_info["height"]

    # Background detection worker thread
    def detection_worker():
        nonlocal model
        with mss.mss() as sct:
            mon = sct.monitors[monitor_idx]
            local_fps = 0.0

            while not quit_event.is_set():
                t0 = time.time()
                shot = np.array(sct.grab(mon))
                frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

                curr_conf = state["conf"]
                preds = model.predict(
                    frame,
                    conf=curr_conf,
                    imgsz=imgsz,
                    device=device,
                    verbose=False,
                )
                res = preds[0]

                boxes = []
                centers = []
                polygons = []

                has_masks = res.masks is not None and len(res.masks) > 0
                has_boxes = res.boxes is not None and len(res.boxes) > 0

                if has_boxes:
                    for idx, b in enumerate(res.boxes):
                        cls_id = int(b.cls[0].item())
                        score = float(b.conf[0].item())
                        if cls_id == target_class_id or class_names.get(cls_id) == "gold_mine":
                            xyxy = b.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = xyxy
                            boxes.append((int(x1), int(y1), int(x2), int(y2), score))
                            centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

                            if has_masks and idx < len(res.masks.xy):
                                poly = res.masks.xy[idx]
                                if len(poly) > 0:
                                    polygons.append(poly)

                # Compute closest drop points on the imaginary perimeter (box-to-perimeter exact minimum)
                drop_data = calculate_drop_points(boxes=boxes, centers=centers, perimeter=DEFAULT_PERIMETER)

                # Snapshot saving
                if snapshot_trigger.is_set():
                    snapshot_trigger.clear()
                    ts = int(time.time())
                    cnt = state["snapshot_count"] + 1
                    sname = f"overlay_snap_{cnt:04d}_{len(boxes)}mines_{ts}.jpg"
                    spath = os.path.join(save_dir, sname)

                    annotated_shot = generate_heatmap(frame, centers, radius=45, sigma=20, blend_only_hot=True)

                    # Draw imaginary perimeter lines on snapshot
                    for i in range(len(DEFAULT_PERIMETER) - 1):
                        p1 = (int(DEFAULT_PERIMETER[i][0]), int(DEFAULT_PERIMETER[i][1]))
                        p2 = (int(DEFAULT_PERIMETER[i + 1][0]), int(DEFAULT_PERIMETER[i + 1][1]))
                        cv2.line(annotated_shot, p1, p2, (255, 255, 0), 2)

                    # Draw drop points and perpendicular guidelines
                    for d in drop_data:
                        m_pt = d.get("nearest_mine_point", d["mine_center"])
                        mx, my = int(m_pt[0]), int(m_pt[1])
                        qx, qy = int(d["drop_point"][0]), int(d["drop_point"][1])
                        cv2.line(annotated_shot, (mx, my), (qx, qy), (0, 255, 100), 1)
                        # Highlight dot
                        cv2.circle(annotated_shot, (qx, qy), 7, (0, 255, 100), -1)
                        cv2.circle(annotated_shot, (qx, qy), 9, (255, 255, 255), 1)

                    # Draw bounding boxes & labels
                    for x1, y1, x2, y2, sc in boxes:
                        cv2.rectangle(annotated_shot, (x1, y1), (x2, y2), (0, 215, 255), 2)
                        cv2.putText(
                            annotated_shot,
                            f"Gold Mine {sc:.2f}",
                            (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 215, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    cv2.imwrite(spath, annotated_shot)
                    state["snapshot_count"] = cnt
                    state["last_snapshot_msg"] = sname
                    state["last_snapshot_time"] = time.time()
                    print(f"[+] Saved snapshot with drop points: {spath}")
                    play_beep()

                dt = max(time.time() - t0, 1e-6)
                local_fps = 0.9 * local_fps + 0.1 * (1.0 / dt) if local_fps > 0 else (1.0 / dt)
                state["fps"] = local_fps

                payload = {
                    "boxes": boxes,
                    "centers": centers,
                    "polygons": polygons,
                    "drop_data": drop_data,
                }
                if results_queue.full():
                    try:
                        results_queue.get_nowait()
                    except queue.Empty:
                        pass
                results_queue.put(payload)

    # Global keyboard listener
    def on_press(key):
        try:
            if key in (keyboard.Key.f8, keyboard.Key.ctrl_r):
                snapshot_trigger.set()
            elif key in (keyboard.Key.esc, keyboard.Key.f12):
                quit_event.set()
            elif hasattr(key, "char") and key.char is not None:
                c = key.char.lower()
                if c == "s":
                    snapshot_trigger.set()
                elif c == "d":
                    state["show_drop_points"] = not state["show_drop_points"]
                    print(f"[INFO] Drop Points & Perimeter: {'ON' if state['show_drop_points'] else 'OFF'}")
                elif c == "h":
                    state["show_heatmap"] = not state["show_heatmap"]
                    print(f"[INFO] Heatmap: {'ON' if state['show_heatmap'] else 'OFF'}")
                elif c == "b":
                    state["show_boxes"] = not state["show_boxes"]
                    print(f"[INFO] Bounding Boxes: {'ON' if state['show_boxes'] else 'OFF'}")
                elif c == "m":
                    state["show_masks"] = not state["show_masks"]
                    print(f"[INFO] Masks: {'ON' if state['show_masks'] else 'OFF'}")
                elif c in ("+", "="):
                    state["conf"] = min(0.95, round(state["conf"] + 0.05, 2))
                    print(f"[INFO] Confidence: {state['conf']:.2f}")
                elif c in ("-", "_"):
                    state["conf"] = max(0.05, round(state["conf"] - 0.05, 2))
                    print(f"[INFO] Confidence: {state['conf']:.2f}")
                elif c == "q":
                    quit_event.set()
        except Exception:
            pass

    if HAS_PYNPUT:
        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
    else:
        listener = None

    worker_thread = threading.Thread(target=detection_worker, daemon=True)
    worker_thread.start()

    # Create Tkinter overlay window
    root = tk.Tk()
    setup_transparent_window(root, mon_left, mon_top, mon_width, mon_height)

    canvas = tk.Canvas(
        root,
        width=mon_width,
        height=mon_height,
        bg="black",
        highlightthickness=0,
    )
    canvas.pack(fill="both", expand=True)

    current_data = {"boxes": [], "centers": [], "polygons": [], "drop_data": []}

    def update_overlay():
        if quit_event.is_set():
            root.destroy()
            return

        try:
            nonlocal current_data
            while not results_queue.empty():
                current_data = results_queue.get_nowait()
        except Exception:
            pass

        canvas.delete("all")

        boxes = current_data.get("boxes", [])
        centers = current_data.get("centers", [])
        polygons = current_data.get("polygons", [])
        drop_data = current_data.get("drop_data", [])

        # 1. Draw Imaginary Perimeter Line & Closest Drop Points
        if state["show_drop_points"]:
            # Draw the imaginary boundary connecting line (Top -> Left -> Bottom -> Right -> Top)
            for i in range(len(DEFAULT_PERIMETER) - 1):
                p1 = DEFAULT_PERIMETER[i]
                p2 = DEFAULT_PERIMETER[i + 1]
                canvas.create_line(
                    p1[0], p1[1], p2[0], p2[1],
                    fill="#00E5FF",
                    dash=(8, 6),
                    width=2,
                )

            # Draw perpendicular trajectory lines and highlight dots
            for d in drop_data:
                mx, my = d.get("nearest_mine_point", d["mine_center"])
                qx, qy = d["drop_point"]
                dist = d["distance"]

                # Thin connecting guideline from gold mine boundary to perimeter drop point
                canvas.create_line(
                    mx, my, qx, qy,
                    fill="#00FF66",
                    dash=(3, 3),
                    width=1,
                )

                # Outer glowing halo ring
                canvas.create_oval(
                    qx - 8, qy - 8, qx + 8, qy + 8,
                    outline="#00FF66",
                    fill="",
                    width=2,
                )
                # Small highlight core dot
                canvas.create_oval(
                    qx - 4, qy - 4, qx + 4, qy + 4,
                    fill="#FFFFFF",
                    outline="#00FF66",
                    width=1,
                )

        # 2. Draw Heatmap (Concentric glowing rings centered on buildings)
        if state["show_heatmap"]:
            for cx, cy in centers:
                canvas.create_oval(
                    cx - 48, cy - 48, cx + 48, cy + 48,
                    outline="#FF4500", width=3,
                )
                canvas.create_oval(
                    cx - 30, cy - 30, cx + 30, cy + 30,
                    outline="#FF8C00", width=2,
                )
                canvas.create_oval(
                    cx - 14, cy - 14, cx + 14, cy + 14,
                    outline="#FFD700", width=2,
                )

        # 3. Draw Segmentation Masks
        if state["show_masks"] and polygons:
            for poly in polygons:
                pts = [coord for pt in poly for coord in pt]
                if len(pts) >= 6:
                    canvas.create_polygon(
                        pts,
                        outline="#00FFFF",
                        fill="",
                        width=2,
                    )

        # 4. Draw Bounding Boxes & Badges
        if state["show_boxes"]:
            for x1, y1, x2, y2, sc in boxes:
                canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline="#FFD700",
                    width=2,
                )
                label_text = f"Gold Mine {sc:.2f}"
                canvas.create_rectangle(
                    x1, max(y1 - 22, 0), x1 + 120, max(y1 - 2, 20),
                    fill="#1A1A1A",
                    outline="#FFD700",
                    width=1,
                )
                canvas.create_text(
                    x1 + 60, max(y1 - 12, 10),
                    text=label_text,
                    fill="#FFD700",
                    font=("Segoe UI", 9, "bold"),
                )

        # 5. Top Status HUD Bar
        now = time.time()
        if now - state["last_snapshot_time"] < 1.2:
            canvas.create_rectangle(15, 12, 540, 50, fill="#006400", outline="#00FF00", width=2)
            canvas.create_text(
                275, 31,
                text=f"[SAVED] {state['last_snapshot_msg']}",
                fill="#FFFFFF",
                font=("Segoe UI", 10, "bold"),
            )
        else:
            h_stat = "ON" if state["show_heatmap"] else "OFF"
            b_stat = "ON" if state["show_boxes"] else "OFF"
            d_stat = "ON" if state["show_drop_points"] else "OFF"

            hud_str = (
                f"Gold Mines: {len(boxes)}  |  Conf: {state['conf']:.2f} (+/-)  |  FPS: {state['fps']:.1f}  |  "
                f"[D] DropPts: {d_stat}  |  [H] Heat: {h_stat}  |  [B] Box: {b_stat}  |  [F8] Snapshot  |  [Q] Exit"
            )
            canvas.create_rectangle(15, 12, 860, 48, fill="#1A1A1A", outline="#FFD700", width=1)
            canvas.create_text(
                435, 30,
                text=hud_str,
                fill="#FFD700",
                font=("Segoe UI", 9, "bold"),
            )

        root.after(16, update_overlay)

    root.after(16, update_overlay)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        quit_event.set()
        if listener:
            listener.stop()
        print("Overlay closed.")


# ==============================================================================
# Legacy Windowed Preview Mode (if user passes --window)
# ==============================================================================

def run_windowed_detection(
    model_path: str,
    conf_threshold: float = 0.45,
    imgsz: int = 800,
    monitor_idx: int = 1,
    device: str | None = None,
    save_dir: str | None = None,
):
    """Run detection in a standalone OpenCV preview window with drop points."""
    if save_dir is None:
        save_dir = get_default_save_dir()
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    model = YOLO(model_path)
    window_name = "CoC Gold Mine Live Detection Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 620)

    show_heatmap = True
    show_boxes = True
    show_drop = True
    conf = conf_threshold

    with mss.mss() as sct:
        if monitor_idx >= len(sct.monitors):
            monitor_idx = 1
        mon = sct.monitors[monitor_idx]

        while True:
            shot = np.array(sct.grab(mon))
            frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

            res = model.predict(frame, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
            boxes = []
            centers = []

            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_id = int(b.cls[0].item())
                    score = float(b.conf[0].item())
                    if cls_id == 0:
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        boxes.append((int(x1), int(y1), int(x2), int(y2), score))
                        centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

            drop_data = calculate_drop_points(centers, DEFAULT_PERIMETER)

            if show_heatmap and centers:
                display = generate_heatmap(frame, centers, radius=45, sigma=20, blend_only_hot=True)
            else:
                display = frame.copy()

            if show_drop:
                for i in range(len(DEFAULT_PERIMETER) - 1):
                    p1 = (int(DEFAULT_PERIMETER[i][0]), int(DEFAULT_PERIMETER[i][1]))
                    p2 = (int(DEFAULT_PERIMETER[i + 1][0]), int(DEFAULT_PERIMETER[i + 1][1]))
                    cv2.line(display, p1, p2, (255, 255, 0), 2)

                for d in drop_data:
                    cx, cy = int(d["mine_center"][0]), int(d["mine_center"][1])
                    qx, qy = int(d["drop_point"][0]), int(d["drop_point"][1])
                    cv2.line(display, (cx, cy), (qx, qy), (0, 255, 100), 1)
                    cv2.circle(display, (qx, qy), 7, (0, 255, 100), -1)

            if show_boxes:
                for x1, y1, x2, y2, sc in boxes:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 215, 255), 2)
                    cv2.putText(display, f"Gold Mine {sc:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in (ord("d"), ord("D")):
                show_drop = not show_drop
            elif key in (ord("h"), ord("H")):
                show_heatmap = not show_heatmap
            elif key in (ord("b"), ord("B")):
                show_boxes = not show_boxes

    cv2.destroyAllWindows()


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Live CoC Gold Mine Detection & Screen Overlay")
    parser.add_argument("--model", default=None, help="Path to model weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (0.05 - 0.95)")
    parser.add_argument("--imgsz", type=int, default=800, help="Inference image resolution")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index to capture (1 = primary)")
    parser.add_argument("--device", default=None, help="Inference device: '0', 'cpu', etc.")
    parser.add_argument("--save-dir", default=None, help="Directory to save detection snapshot images")
    parser.add_argument("--window", action="store_true", help="Run legacy separate window instead of transparent overlay")
    args = parser.parse_args()

    model_path = args.model if args.model else find_model_weights()
    if not os.path.exists(model_path):
        print(f"[ERROR] Model weights not found at '{model_path}'.")
        print("Please train a model first using: python src/train.py")
        sys.exit(1)

    if args.window:
        run_windowed_detection(
            model_path=model_path,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            monitor_idx=args.monitor,
            device=args.device,
            save_dir=args.save_dir,
        )
    else:
        run_live_overlay(
            model_path=model_path,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            monitor_idx=args.monitor,
            device=args.device,
            save_dir=args.save_dir,
        )


if __name__ == "__main__":
    main()
