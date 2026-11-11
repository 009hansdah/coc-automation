"""
Coordinate picker for mouse/keyboard automation.

Install:  pip install pyautogui pynput

Usage:
  1. Run:  python coord_picker.py
  2. Move the mouse over the spot you want in your software.
  3. Press F8 to record that position (give it a name when asked at the end,
     or just note the order).  Press Esc to finish.
  4. Positions are printed and saved to coords.json.

The live position is shown in the terminal while you move the mouse.
"""
import json
import sys
import pyautogui
from pynput import keyboard

points = []


def on_press(key):
    if key == keyboard.Key.f8:
        x, y = pyautogui.position()
        points.append({"index": len(points) + 1, "x": x, "y": y})
        print(f"\nRecorded #{len(points)}: x={x}, y={y}")
    elif key == keyboard.Key.esc:
        return False  # stop listener


def main():
    print("Move the mouse to a target and press F8 to record. Esc to finish.\n")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    try:
        while listener.running:
            x, y = pyautogui.position()
            sys.stdout.write(f"\rCurrent position: x={x:<5} y={y:<5}")
            sys.stdout.flush()
            listener.join(0.05)
    except KeyboardInterrupt:
        pass

    print("\n\nRecorded points:")
    for p in points:
        print(f"  #{p['index']}: ({p['x']}, {p['y']})")

    with open("coords.json", "w") as f:
        json.dump(points, f, indent=2)
    print("\nSaved to coords.json")


if __name__ == "__main__":
    main()