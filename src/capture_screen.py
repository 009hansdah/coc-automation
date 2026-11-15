"""
Gameplay screen capture helper for Clash of Clans.
Captures screenshots to build the dataset for training building detection models.

Features:
  - GLOBAL HOTKEYS (works even when Clash of Clans / Emulator window is focused!)
    * [F8] or [Right Ctrl] : Global capture & save
    * [S] or [Space]       : Capture & save (works globally or in preview window)
    * [A]                  : Toggle Auto-capture mode (saves a frame every 3s)
    * [Esc] or [F12]       : Quit
  - Instant Audio Beep feedback on save so you know it captured without looking away.
  - Multi-backend capture: High-speed MSS with automatic PyAutoGUI/PIL fallback.
  - Robust file saving: Never overwrites existing images in dataset/raw/.
"""

import argparse
import os
import sys
import threading
import time
import cv2
import numpy as np

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

# Global hotkey listener
from pynput import keyboard


def get_save_dir() -> str:
    """Return the absolute path to dataset/raw, independent of working directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(project_root, "dataset", "raw")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def get_next_image_index(save_dir: str) -> int:
    """Find the highest existing image number to prevent overwriting."""
    max_index = 0
    if os.path.exists(save_dir):
        for name in os.listdir(save_dir):
            if name.lower().startswith("image_") and name.lower().endswith((".jpg", ".jpeg", ".png")):
                base = os.path.splitext(name)[0]
                parts = base.split("_")
                for part in parts[1:]:
                    if part.isdigit():
                        max_index = max(max_index, int(part))
    return max_index + 1


def grab_screen(sct, monitor):
    """Grab a frame using MSS, falling back to PyAutoGUI if needed."""
    if sct is not None:
        try:
            shot = np.array(sct.grab(monitor))
            return cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        except Exception:
            pass

    # Fallback to PyAutoGUI
    try:
        import pyautogui
        shot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    except Exception:
        from PIL import ImageGrab
        shot = ImageGrab.grab()
        return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description="CoC Screen Capture Tool")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index to capture (1 = primary)")
    parser.add_argument("--no-preview", action="store_true", help="Run in headless background mode without preview window")
    parser.add_argument("--auto", type=float, default=0.0, help="Auto-capture interval in seconds (0 = disabled)")
    args = parser.parse_args()

    save_dir = get_save_dir()
    image_index = get_next_image_index(save_dir)

    # Threading events for global hotkeys
    save_trigger = threading.Event()
    quit_trigger = threading.Event()
    auto_capture_active = args.auto > 0
    auto_interval = args.auto if args.auto > 0 else 5.0

    def on_press(key):
        nonlocal auto_capture_active
        try:
            # Global capture hotkeys: F8 or Right Ctrl
            if key == keyboard.Key.f8 or key == keyboard.Key.ctrl_r:
                save_trigger.set()
            elif key == keyboard.Key.esc or key == keyboard.Key.f12:
                quit_trigger.set()
            elif hasattr(key, "char") and key.char is not None:
                char = key.char.lower()
                if char == "s":
                    save_trigger.set()
                elif char == "a":
                    auto_capture_active = not auto_capture_active
                    state = "ENABLED (every 3s)" if auto_capture_active else "DISABLED"
                    print(f"\n[INFO] Auto-capture {state}")
                elif char == "q":
                    quit_trigger.set()
        except Exception:
            pass

    # Start global keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

    # Initialize MSS screen capture
    import mss
    sct = mss.mss()
    if args.monitor < len(sct.monitors):
        monitor = sct.monitors[args.monitor]
    else:
        monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]

    print("=" * 65)
    print("      Clash of Clans - Screen Capture Tool")
    print("=" * 65)
    print(f"Target folder   : {save_dir}")
    print(f"Next image ID   : #{image_index:05d}")
    print(f"Monitor         : {monitor.get('width', 'auto')}x{monitor.get('height', 'auto')}")
    print("-" * 65)
    print("GLOBAL HOTKEYS (Works even while playing inside the game!):")
    print("  [F8] or [Right Ctrl] : Save screenshot")
    print("  [S] or [Space]       : Save screenshot")
    print("  [A]                  : Toggle auto-capture (every 3s)")
    print("  [Esc] or [F12]       : Quit capture tool")
    print("=" * 65)

    window_name = "CoC Dataset Capture (Press F8 or S to Save)"
    if not args.no_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Resize preview window to compact size so it doesn't cover the entire screen
        cv2.resizeWindow(window_name, 960, 540)

    last_save_time = 0.0
    last_auto_save_time = time.time()
    last_saved_filename = ""
    saved_count_session = 0

    try:
        while not quit_trigger.is_set():
            t0 = time.time()
            frame = grab_screen(sct, monitor)
            if frame is None:
                time.sleep(0.05)
                continue

            # Check auto-capture
            if auto_capture_active and (time.time() - last_auto_save_time >= auto_interval):
                save_trigger.set()
                last_auto_save_time = time.time()

            # Handle save trigger
            if save_trigger.is_set():
                save_trigger.clear()

                filename = f"image_{image_index:05d}.jpg"
                filepath = os.path.join(save_dir, filename)

                # Ensure we don't overwrite if file somehow exists
                while os.path.exists(filepath):
                    image_index += 1
                    filename = f"image_{image_index:05d}.jpg"
                    filepath = os.path.join(save_dir, filename)

                # Save the clean frame (without any UI annotations)
                success = cv2.imwrite(filepath, frame)

                if success and os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    saved_count_session += 1
                    last_saved_filename = filename
                    last_save_time = time.time()
                    file_kb = os.path.getsize(filepath) // 1024
                    print(f"[+] Saved #{image_index:05d}: {filepath} ({frame.shape[1]}x{frame.shape[0]}, {file_kb} KB)")
                    play_beep()
                    image_index += 1
                else:
                    print(f"[ERROR] Failed to write image: {filepath}")

            if not args.no_preview:
                display_frame = frame.copy()

                # Visual flash message if saved within last 1 second
                if time.time() - last_save_time < 1.0:
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 255, 0), 10)
                    cv2.putText(
                        display_frame,
                        f"SAVED: {last_saved_filename}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                        cv2.LINE_AA,
                    )
                else:
                    fps = 1.0 / max(time.time() - t0, 1e-6)
                    auto_status = "[AUTO: ON]" if auto_capture_active else "[AUTO: OFF]"
                    hud_msg = f"Saved: {saved_count_session} (Total: {image_index - 1}) | {auto_status} | FPS: {fps:.1f} | [F8/S] Save [Q] Quit"
                    cv2.rectangle(display_frame, (10, 10), (780, 48), (20, 20, 20), -1)
                    cv2.putText(
                        display_frame,
                        hud_msg,
                        (20, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("s"), ord("S"), 32):  # 's' or Space in window
                    save_trigger.set()
                elif key in (ord("q"), ord("Q"), 27):  # 'q' or Esc in window
                    break
                elif key in (ord("a"), ord("A")):
                    auto_capture_active = not auto_capture_active
                    print(f"[INFO] Auto-capture {'ENABLED' if auto_capture_active else 'DISABLED'}")
            else:
                # Headless sleep to yield CPU
                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    finally:
        listener.stop()
        if not args.no_preview:
            cv2.destroyAllWindows()
        sct.close()

    print("=" * 65)
    print(f"Capture session ended.")
    print(f"Images saved this session: {saved_count_session}")
    print(f"Total images in {save_dir}: {len([f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.png'))])}")
    print("=" * 65)


if __name__ == "__main__":
    main()
