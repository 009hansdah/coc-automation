"""
Main entry point for Clash of Clans Gold Mine Detection & Heatmap.
Provides quick access to screen capture, training, evaluation, and live detection.
"""

import argparse
import sys
import os

# Add src to sys.path so relative imports work smoothly
sys.path.insert(0, os.path.dirname(__file__))


def print_menu():
    print("=" * 60)
    print("   Clash of Clans - Gold Mine Detection System")
    print("=" * 60)
    print("  1. Capture screenshots for dataset   (src/capture_screen.py)")
    print("  2. Train Gold Mine detector          (src/train.py)")
    print("  3. Evaluate trained model            (src/evaluate.py)")
    print("  4. Start Live Detection & Heatmap    (src/detect_live.py)")
    print("  5. Run Pipeline Self-Test Check      (src/test_pipeline.py)")
    print("  0. Exit")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CoC Gold Mine Detection CLI")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["capture", "train", "eval", "evaluate", "live", "detect", "test"],
        help="Action to perform directly without menu",
    )
    args, unknown = parser.parse_known_args()

    action = args.action

    if action is None:
        print_menu()
        choice = input("Select an option [0-5]: ").strip()
        mapping = {
            "1": "capture",
            "2": "train",
            "3": "eval",
            "4": "live",
            "5": "test",
            "0": "exit",
        }
        action = mapping.get(choice)
        if not action or action == "exit":
            print("Exiting.")
            return

    if action == "capture":
        import capture_screen
        capture_screen.main()
    elif action == "train":
        import train
        train.main()
    elif action in ("eval", "evaluate"):
        import evaluate
        evaluate.main()
    elif action in ("live", "detect"):
        import detect_live
        detect_live.main()
    elif action == "test":
        import test_pipeline
        test_pipeline.main()


if __name__ == "__main__":
    main()

