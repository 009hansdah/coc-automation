"""
Smart Gold Mine attack automation with live transparent HUD overlay.

Detects Gold Mines via YOLO, computes optimal drop points on the deployment
perimeter, deploys troops by holding left-click at each point, and renders
a click-through overlay showing bounding boxes, perimeter, and drop points.

Usage:
  python src/smart_attack.py
  - Press RIGHT CTRL once  -> start looping attacks
  - Press RIGHT CTRL again -> stop
  - Press V                -> toggle overlay visibility
  - Press ESC              -> quit
  - Emergency stop: slam mouse into any screen corner (pyautogui failsafe)
"""

import ctypes
import math
import os
import queue
import sys
import threading
import time
import tkinter as tk

import cv2
import mss
import numpy as np
import pyautogui
import torch
from pynput import keyboard
from ultralytics import YOLO

# Ensure src/ is on path for local imports
sys.path.insert(0, os.path.dirname(__file__))
from drop_calculator import DEFAULT_PERIMETER, calculate_drop_points

# ---------------------------------------------------------------- settings
HOTKEY = keyboard.Key.ctrl_r
QUIT_KEY = keyboard.Key.esc
STEP_DELAY = 0.3
LOOP_DELAY = 3.0
MOVE_DURATION = 0.1
HOLD_SECONDS = 1.0          # how long to hold left-click at each drop point
DEDUP_RADIUS = 15            # merge drop points closer than this (px)

MODEL_PATH = "models/gold_mine_best.pt"
CONF_THRESHOLD = 0.65
IMGSZ = 960

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

# ---------------------------------------------------------------- shared state
running = threading.Event()
quit_flag = threading.Event()

overlay_state = {
    "show_overlay": True,     # V to toggle
    "boxes": [],              # [(x1,y1,x2,y2,score), ...]
    "drop_data": [],          # list of dicts from calculate_drop_points
    "phase": "idle",          # idle | navigating | detecting | deploying | exiting
    "loop_count": 0,
    "deploy_index": 0,        # which drop point is being deployed
    "deploy_total": 0,
}
overlay_lock = threading.Lock()
overlay_queue = queue.Queue(maxsize=1)


# ---------------------------------------------------------------- engine
def interruptible_sleep(seconds):
    """Sleep in small slices so stopping with the hotkey is responsive."""
    end = time.time() + seconds
    while time.time() < end:
        if not running.is_set() or quit_flag.is_set():
            return False
        time.sleep(0.05)
    return True


def do_action(action):
    t = action["type"]
    if t == "click":
        pyautogui.moveTo(action["x"], action["y"], duration=MOVE_DURATION)
        pyautogui.click(button="left")
    elif t == "wait":
        return interruptible_sleep(action["seconds"])
    elif t == "key":
        pyautogui.press(action["key"])
    else:
        print(f"Unknown action type: {t}")
    return True


def detect_and_get_drop_points(model):
    """Capture screen, run YOLO, return (boxes, drop_data, unique_points)."""
    with mss.mss() as sct:
        mon = sct.monitors[1]
        shot = np.array(sct.grab(mon))
        frame = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)

    preds = model.predict(frame, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)
    res = preds[0]

    boxes = []
    if res.boxes is not None and len(res.boxes) > 0:
        class_names = getattr(model, "names", {0: "gold_mine"})
        for b in res.boxes:
            cls_id = int(b.cls[0].item())
            if cls_id == 0 or class_names.get(cls_id) == "gold_mine":
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                boxes.append((int(x1), int(y1), int(x2), int(y2), float(b.conf[0].item())))

    if not boxes:
        return boxes, [], []

    drop_data = calculate_drop_points(boxes=boxes, perimeter=DEFAULT_PERIMETER)

    # Deduplicate close drop points
    unique_points = []
    for d in drop_data:
        qx, qy = d["drop_point"]
        if not any(math.hypot(qx - px, qy - py) < DEDUP_RADIUS for px, py in unique_points):
            unique_points.append((int(qx), int(qy)))

    return boxes, drop_data, unique_points


def hold_click_at(x, y, seconds):
    """Move to (x, y), press and hold left-click for `seconds`, then release."""
    pyautogui.moveTo(x, y, duration=MOVE_DURATION)
    pyautogui.mouseDown(button="left")
    try:
        interruptible_sleep(seconds)
    finally:
        pyautogui.mouseUp(button="left")


def set_phase(phase):
    with overlay_lock:
        overlay_state["phase"] = phase


def worker(model):
    iteration = 0
    while not quit_flag.is_set():
        if not running.is_set():
            set_phase("idle")
            time.sleep(0.1)
            continue

        iteration += 1
        with overlay_lock:
            overlay_state["loop_count"] = iteration
        print(f"\n{'='*50}")
        print(f"--- Loop {iteration} ---")
        print(f"{'='*50}")

        # --- Phase 1: Navigate to attack ---
        set_phase("navigating")
        pre_actions = [
            {"type": "click", "name": "battle",       "x": 105,  "y": 978},
            {"type": "click", "name": "find a match",  "x": 267,  "y": 759},
            {"type": "click", "name": "attack",        "x": 1638, "y": 921},
            {"type": "wait",  "seconds": 5},
            {"type": "click", "name": "troop",         "x": 356,  "y": 996},
        ]
        aborted = False
        for action in pre_actions:
            if not running.is_set() or quit_flag.is_set():
                aborted = True
                break
            label = action.get("name") or action["type"]
            print(f"  {label}")
            if do_action(action) is False:
                aborted = True
                break
            if action["type"] != "wait":
                interruptible_sleep(STEP_DELAY)

        if aborted:
            continue

        # --- Phase 2: Detect gold mines and deploy at drop points ---
        set_phase("detecting")
        print("  [DETECT] Scanning for Gold Mines...")
        boxes, drop_data, unique_points = detect_and_get_drop_points(model)

        # Push detection results to overlay
        with overlay_lock:
            overlay_state["boxes"] = boxes
            overlay_state["drop_data"] = drop_data
            overlay_state["deploy_total"] = len(unique_points)
            overlay_state["deploy_index"] = 0
        # Signal overlay to redraw
        try:
            overlay_queue.get_nowait()
        except queue.Empty:
            pass
        overlay_queue.put(True)

        print(f"  [DETECT] Found {len(boxes)} Gold Mine(s), {len(unique_points)} unique drop point(s)")

        if unique_points:
            set_phase("deploying")
            for i, (qx, qy) in enumerate(unique_points, 1):
                if not running.is_set() or quit_flag.is_set():
                    break
                with overlay_lock:
                    overlay_state["deploy_index"] = i
                print(f"  [DEPLOY {i}/{len(unique_points)}] Hold at ({qx}, {qy}) for {HOLD_SECONDS}s")
                hold_click_at(qx, qy, HOLD_SECONDS)
                interruptible_sleep(STEP_DELAY)
        else:
            print("  [DEPLOY] No Gold Mines detected, skipping deployment")

        # --- Phase 3: Wait, surrender, go home ---
        set_phase("exiting")
        post_actions = [
            {"type": "wait",  "seconds": 10},
            {"type": "click", "name": "surrender", "x": 121,  "y": 853},
            {"type": "wait",  "seconds": 0.5},
            {"type": "click", "name": "okay",      "x": 1129, "y": 661},
            {"type": "wait",  "seconds": 0.7},
            {"type": "click", "name": "home",      "x": 971,  "y": 930},
        ]
        for action in post_actions:
            if not running.is_set() or quit_flag.is_set():
                break
            label = action.get("name") or action["type"]
            print(f"  {label}")
            if do_action(action) is False:
                break
            if action["type"] != "wait":
                interruptible_sleep(STEP_DELAY)

        # Clear detection data after round ends
        with overlay_lock:
            overlay_state["boxes"] = []
            overlay_state["drop_data"] = []
            overlay_state["deploy_index"] = 0
            overlay_state["deploy_total"] = 0

        print(f"  [LOOP] Waiting {LOOP_DELAY}s before next attack...")
        set_phase("idle")
        interruptible_sleep(LOOP_DELAY)


# ---------------------------------------------------------------- overlay
def setup_transparent_window(root, left, top, width, height):
    """Configure a Tkinter window as a transparent, topmost, click-through overlay."""
    root.overrideredirect(True)
    root.geometry(f"{width}x{height}+{left}+{top}")
    root.wm_attributes("-topmost", True)
    root.wm_attributes("-transparentcolor", "black")
    root.config(bg="black")
    root.update()

    user32 = ctypes.windll.user32
    hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()

    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOPMOST = 0x00000008

    current_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(
        hwnd, GWL_EXSTYLE,
        current_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST,
    )

    try:
        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
    except Exception:
        pass


def run_overlay(mon_width, mon_height, mon_left, mon_top):
    """Tkinter overlay loop — reads shared state and draws on screen."""
    root = tk.Tk()
    setup_transparent_window(root, mon_left, mon_top, mon_width, mon_height)

    canvas = tk.Canvas(root, width=mon_width, height=mon_height,
                       bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    def update():
        if quit_flag.is_set():
            root.destroy()
            return

        canvas.delete("all")

        with overlay_lock:
            show = overlay_state["show_overlay"]
            boxes = list(overlay_state["boxes"])
            drop_data = list(overlay_state["drop_data"])
            phase = overlay_state["phase"]
            loop_count = overlay_state["loop_count"]
            dep_idx = overlay_state["deploy_index"]
            dep_total = overlay_state["deploy_total"]

        if show:
            # 1. Perimeter boundary
            for i in range(len(DEFAULT_PERIMETER) - 1):
                p1 = DEFAULT_PERIMETER[i]
                p2 = DEFAULT_PERIMETER[i + 1]
                canvas.create_line(
                    p1[0], p1[1], p2[0], p2[1],
                    fill="#00E5FF", dash=(8, 6), width=2,
                )

            # 2. Bounding boxes
            for x1, y1, x2, y2, sc in boxes:
                canvas.create_rectangle(
                    x1, y1, x2, y2, outline="#FFD700", width=2,
                )
                label = f"Gold Mine {sc:.2f}"
                canvas.create_rectangle(
                    x1, max(y1 - 22, 0), x1 + 120, max(y1 - 2, 20),
                    fill="#1A1A1A", outline="#FFD700", width=1,
                )
                canvas.create_text(
                    x1 + 60, max(y1 - 12, 10),
                    text=label, fill="#FFD700",
                    font=("Segoe UI", 9, "bold"),
                )

            # 3. Drop points & guidelines
            for d in drop_data:
                mx, my = d.get("nearest_mine_point", d["mine_center"])
                qx, qy = d["drop_point"]

                canvas.create_line(
                    mx, my, qx, qy,
                    fill="#00FF66", dash=(3, 3), width=1,
                )
                canvas.create_oval(
                    qx - 8, qy - 8, qx + 8, qy + 8,
                    outline="#00FF66", fill="", width=2,
                )
                canvas.create_oval(
                    qx - 4, qy - 4, qx + 4, qy + 4,
                    fill="#FFFFFF", outline="#00FF66", width=1,
                )

            # 4. Status HUD
            phase_labels = {
                "idle": "⏸ IDLE",
                "navigating": "🔄 NAVIGATING",
                "detecting": "🔍 DETECTING",
                "deploying": f"🎯 DEPLOYING {dep_idx}/{dep_total}",
                "exiting": "🏠 EXITING",
            }
            run_status = "RUNNING" if running.is_set() else "STOPPED"
            hud = (
                f"Smart Attack  |  Loop: {loop_count}  |  {run_status}  |  "
                f"{phase_labels.get(phase, phase)}  |  "
                f"Mines: {len(boxes)}  |  Drops: {dep_total}  |  "
                f"[V] Toggle  |  [RCTRL] Start/Stop  |  [ESC] Quit"
            )
            canvas.create_rectangle(15, 12, 920, 48,
                                    fill="#1A1A1A", outline="#FFD700", width=1)
            canvas.create_text(465, 30, text=hud,
                               fill="#FFD700", font=("Segoe UI", 9, "bold"))

        root.after(33, update)  # ~30 FPS overlay refresh

    root.after(33, update)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- hotkeys
def on_press(key):
    if key == HOTKEY:
        if running.is_set():
            running.clear()
            print(">>> STOPPED (press right CTRL to start again)")
        else:
            running.set()
            print(">>> STARTED")
    elif key == QUIT_KEY:
        print(">>> Quitting")
        running.clear()
        quit_flag.set()
        return False
    elif hasattr(key, "char") and key.char is not None:
        if key.char.lower() == "v":
            with overlay_lock:
                overlay_state["show_overlay"] = not overlay_state["show_overlay"]
                vis = "ON" if overlay_state["show_overlay"] else "OFF"
            print(f"[INFO] Overlay: {vis}")


# ---------------------------------------------------------------- main
def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, MODEL_PATH)
    if not os.path.exists(model_path):
        for fallback in ["runs/detect/coc_gold_mine_detector-3/weights/best.pt",
                         "runs/detect/coc_gold_mine_detector/weights/best.pt"]:
            p = os.path.join(project_root, fallback)
            if os.path.exists(p):
                model_path = p
                break

    print("=" * 60)
    print("   CoC Smart Attack — Vision-Guided Troop Deployment")
    print("=" * 60)
    print(f"Model           : {model_path}")
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"Device          : {device}")
    print(f"Hold per point  : {HOLD_SECONDS}s")
    print(f"Confidence      : {CONF_THRESHOLD}")
    print(f"Perimeter       : {len(DEFAULT_PERIMETER) - 1} segments")
    print("-" * 60)
    print("CONTROLS:")
    print("  [Right CTRL]  : Start / Stop attack loop")
    print("  [V]           : Toggle overlay on/off")
    print("  [ESC]         : Quit")
    print("=" * 60)

    model = YOLO(model_path)

    # Get monitor geometry for overlay
    with mss.mss() as sct:
        mon = sct.monitors[1]
        mon_left, mon_top = mon["left"], mon["top"]
        mon_w, mon_h = mon["width"], mon["height"]

    # Start worker thread
    threading.Thread(target=worker, args=(model,), daemon=True).start()

    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    # Run overlay on main thread (Tkinter requires main thread)
    run_overlay(mon_w, mon_h, mon_left, mon_top)

    # Cleanup
    quit_flag.set()
    listener.stop()
    print("Smart Attack exited.")


if __name__ == "__main__":
    main()
