"""
Mouse/keyboard automation loop with a global hotkey.

Install:  pip install pyautogui pynput

Usage:
  python auto_actions.py
  - Press RIGHT CTRL once  -> start looping the actions
  - Press RIGHT CTRL again -> stop (finishes/aborts the current step quickly)
  - Press ESC              -> quit the script completely
  - Emergency stop: slam the mouse into any screen corner (pyautogui failsafe)

The script listens for hotkeys globally, so it works while running in the
background.
"""
import math
import threading
import time

import pyautogui
from pynput import keyboard

# ---------------------------------------------------------------- settings
HOTKEY = keyboard.Key.ctrl_r      # right CTRL toggles start/stop
QUIT_KEY = keyboard.Key.esc       # quit the script
STEP_DELAY = 0.3                  # pause after each action (seconds)
LOOP_DELAY = 3.0                  # pause between loop iterations (seconds)
MOVE_DURATION = 0.1               # time the cursor takes to glide to a point
SCROLL_AMOUNT = 1                 # +1 = scroll up per notch (zooms out); use -1 to scroll down instead
SCROLL_DELAY = 0.3                # pause between scroll notches

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

# ---------------------------------------------------------------- actions
# Each action is a dict. Supported types:
#   click  : {"type": "click", "x":..., "y":..., "name": "..."}
#   wait   : {"type": "wait", "seconds": N}
#   scroll : {"type": "scroll", "x":..., "y":..., "times": N}
#   key    : {"type": "key", "key": "esc"}            (single key press)
#   hotkey : {"type": "hotkey", "keys": ["ctrl", "a"]}
#   type   : {"type": "type", "text": "hello"}
#   drag   : {"type": "drag", "points": [...], "hold": 2, "total_seconds": 9}
#            presses left button at the first point, holds, glides in straight
#            lines through the rest, releases at the last point.
#            total_seconds is timed from button press to release (hold included).

# Drag route: top -> left -> bottom -> right -> top
DRAG_POINTS = [
    (859, 0),     # 5 top    (press + hold here)
    (79, 521),    # 6 left
    (971, 1079),   # 7 bottom
    (1737, 510),   # 8 right
    (1062, 0),      # 9 top    (release here)
]

ACTIONS = [
    {"type": "click", "name": "battle",        "x": 105,  "y": 978},
    {"type": "click", "name": "find a match",  "x": 267,  "y": 759},
    {"type": "click", "name": "attack",        "x": 1638, "y": 921},
    {"type": "wait", "seconds": 5},
    {"type": "click", "name": "troop", "x": 356, "y": 996},
    # {"type": "scroll", "name": "center of the map (zoom out)",
    #  "x": 964, "y": 499, "times": 3},
    # Press the "1" key (number row above QWERTY) once, before the drag sets
    {"type": "key", "name": "press 1", "key": "1"},
    # Run 1 of the drag set: 9 seconds total
    {"type": "drag", "name": "drag run 1 (9s)", "points": DRAG_POINTS,
     "hold": 0.75, "total_seconds": 11.5},
    # Run 2 of the drag set: 4 seconds total
    # {"type": "click", "name": "troop2",        "x": 475, "y": 997},
    {"type": "drag", "name": "drag run 2 (4s)", "points": DRAG_POINTS,
     "hold": 0.75, "total_seconds": 4},
    # ---- add further actions here ----
    {"type": "wait", "seconds": 30},
    {"type": "click", "name": "home",        "x": 958, "y": 930},
]

# ---------------------------------------------------------------- engine
running = threading.Event()   # set  = loop is active
quit_flag = threading.Event()


def interruptible_sleep(seconds):
    """Sleep in small slices so stopping with the hotkey is responsive."""
    end = time.time() + seconds
    while time.time() < end:
        if not running.is_set() or quit_flag.is_set():
            return False
        time.sleep(0.05)
    return True


def drag_path(points, hold_seconds, total_seconds):
    """Press left button at points[0], hold, then glide in straight lines
    through the remaining points and release at the last one.
    total_seconds runs from button press to release (hold included)."""
    move_seconds = max(total_seconds - hold_seconds, 0.1)
    segments = list(zip(points[:-1], points[1:]))
    lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in segments]
    total_len = sum(lengths) or 1

    pyautogui.moveTo(points[0][0], points[0][1], duration=MOVE_DURATION)
    pyautogui.mouseDown(button="left")
    try:
        if not interruptible_sleep(hold_seconds):
            return False
        for (a, b), length in zip(segments, lengths):
            seg_time = move_seconds * length / total_len  # time share by distance
            start = time.time()
            while True:
                if not running.is_set() or quit_flag.is_set():
                    return False
                progress = (time.time() - start) / seg_time if seg_time > 0 else 1
                if progress >= 1:
                    break
                x = a[0] + (b[0] - a[0]) * progress
                y = a[1] + (b[1] - a[1]) * progress
                pyautogui.moveTo(x, y, _pause=False)
                time.sleep(0.01)
            pyautogui.moveTo(b[0], b[1], _pause=False)
        return True
    finally:
        pyautogui.mouseUp(button="left")  # always release the button


def do_action(action):
    t = action["type"]
    if t == "click":
        pyautogui.moveTo(action["x"], action["y"], duration=MOVE_DURATION)
        pyautogui.click(button="left")
    elif t == "wait":
        return interruptible_sleep(action["seconds"])
    elif t == "scroll":
        pyautogui.moveTo(action["x"], action["y"], duration=MOVE_DURATION)
        for _ in range(action.get("times", 1)):
            if not running.is_set():
                return False
            pyautogui.scroll(SCROLL_AMOUNT)
            time.sleep(SCROLL_DELAY)
    elif t == "drag":
        return drag_path(action["points"], action.get("hold", 0),
                         action["total_seconds"])
    elif t == "key":
        pyautogui.press(action["key"])
    elif t == "hotkey":
        pyautogui.hotkey(*action["keys"])
    elif t == "type":
        pyautogui.write(action["text"], interval=0.05)
    else:
        print(f"Unknown action type: {t}")
    return True


def worker():
    iteration = 0
    while not quit_flag.is_set():
        if not running.is_set():
            time.sleep(0.1)
            continue

        iteration += 1
        print(f"--- Loop {iteration} ---")
        for action in ACTIONS:
            if not running.is_set() or quit_flag.is_set():
                break
            label = action.get("name") or action["type"]
            print(f"  {label}")
            try:
                if do_action(action) is False:
                    break
            except pyautogui.FailSafeException:
                print("Failsafe triggered (mouse in corner). Stopping.")
                running.clear()
                break
            if action["type"] != "wait":
                interruptible_sleep(STEP_DELAY)
        else:
            interruptible_sleep(LOOP_DELAY)


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


def main():
    print("Ready. Right CTRL = start/stop, ESC = quit.")
    threading.Thread(target=worker, daemon=True).start()
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == "__main__":
    main()