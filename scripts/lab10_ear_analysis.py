#!/usr/bin/env python3
"""
Lab #10 — EAR Analysis: From 50 Facial Landmarks to Drowsiness Detection.

Deliverable:
    (a) EAR computation for left/right eyes across a frame sequence
    (b) EAR time-series with threshold and blink/drowsy marking
    (c) Comparison: no-glasses vs glasses (measurement error analysis)

Usage:
    python3 scripts/lab10_ear_analysis.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dms_features import compute_ear, PERCLOSTracker
from src.landmarks import FaceLandmarks
from src.landmark_generator import (
    generate_normal_driving,
    generate_drowsy_sequence,
    generate_glasses_sequence,
)

EAR_CLOSED_THRESHOLD = 0.20
BLINK_MIN_FRAMES = 2


def analyze_sequence(
    frames: list[FaceLandmarks],
    label: str,
    fps: int = 60,
) -> dict:
    """Compute EAR stats and detect blinks for a landmark sequence."""
    dt = 1.0 / fps
    perclos_tracker = PERCLOSTracker(window_seconds=60.0, ear_threshold=EAR_CLOSED_THRESHOLD)

    ear_left_series = []
    ear_right_series = []
    ear_avg_series = []
    timestamps = []

    # Blink detection state
    in_blink = False
    blink_start_frame = 0
    blinks = []
    false_closures = []  # glasses-caused false positives

    for i, frame in enumerate(frames):
        ear_l = compute_ear(frame.left_eye())
        ear_r = compute_ear(frame.right_eye())
        ear_avg = (ear_l + ear_r) / 2.0

        ear_left_series.append(ear_l)
        ear_right_series.append(ear_r)
        ear_avg_series.append(ear_avg)
        timestamps.append(frame.timestamp)

        perclos_tracker.update(frame.timestamp, ear_avg)

        # Blink / closure detection
        is_closed = ear_avg < EAR_CLOSED_THRESHOLD
        if is_closed and not in_blink:
            in_blink = True
            blink_start_frame = i
        elif not is_closed and in_blink:
            in_blink = False
            blink_dur_frames = i - blink_start_frame
            if blink_dur_frames >= BLINK_MIN_FRAMES:
                blink_entry = {
                    "start_frame": blink_start_frame,
                    "end_frame": i,
                    "duration_ms": blink_dur_frames * (1000.0 / fps),
                    "min_ear": min(ear_avg_series[blink_start_frame:i]),
                    "timestamp": frames[blink_start_frame].timestamp,
                }
                # Check if any frame in this blink had occluded eye landmarks
                has_occlusion = any(
                    any(frames[f].occluded[j] for j in range(14, 30))
                    for f in range(blink_start_frame, min(i, len(frames)))
                )
                if has_occlusion and frames[0].glasses:
                    false_closures.append(blink_entry)
                else:
                    blinks.append(blink_entry)

    # Final PERCLOS
    perclos_final = perclos_tracker.update(
        timestamps[-1] if timestamps else 0.0,
        ear_avg_series[-1] if ear_avg_series else 0.3,
    )

    n = len(ear_avg_series)
    mean_l = sum(ear_left_series) / n if n else 0
    mean_r = sum(ear_right_series) / n if n else 0
    std_l = (sum((x - mean_l) ** 2 for x in ear_left_series) / n) ** 0.5 if n else 0
    std_r = (sum((x - mean_r) ** 2 for x in ear_right_series) / n) ** 0.5 if n else 0

    return {
        "label": label,
        "total_frames": n,
        "duration_s": timestamps[-1] if timestamps else 0.0,
        "mean_ear_l": mean_l,
        "mean_ear_r": mean_r,
        "std_ear_l": std_l,
        "std_ear_r": std_r,
        "blinks_detected": len(blinks),
        "false_closures": len(false_closures),
        "false_closure_frames": [fc["start_frame"] for fc in false_closures],
        "perclos": perclos_final,
        "ear_avg_series": ear_avg_series,
        "timestamps": timestamps,
        "blinks": blinks,
        "false_closure_details": false_closures,
        "glasses": frames[0].glasses if frames else False,
    }


def print_text_chart(timestamps: list[float], ear_series: list[float], threshold: float, width: int = 70):
    """Print a simple ASCII time-series of EAR values."""
    if not ear_series:
        return

    # Sample at most `width` points
    n = len(ear_series)
    step = max(1, n // width)
    sampled_t = timestamps[::step]
    sampled_ear = ear_series[::step]

    max_ear = max(max(sampled_ear), threshold + 0.05)
    min_ear = min(min(sampled_ear), 0.0)
    ear_range = max_ear - min_ear if max_ear > min_ear else 1.0

    height = 12
    grid = [[' ' for _ in range(len(sampled_ear))] for _ in range(height)]

    # Threshold line
    thresh_row = int((max_ear - threshold) / ear_range * (height - 1))
    thresh_row = max(0, min(height - 1, thresh_row))
    for col in range(len(sampled_ear)):
        grid[thresh_row][col] = '─'

    # Plot EAR
    for col, ear in enumerate(sampled_ear):
        row = int((max_ear - ear) / ear_range * (height - 1))
        row = max(0, min(height - 1, row))
        marker = '●' if ear < threshold else '·'
        grid[row][col] = marker

    # Render
    for row_idx in range(height):
        y_val = max_ear - row_idx * ear_range / (height - 1)
        label = f"{y_val:.2f}" if row_idx % 3 == 0 else "    "
        line = ''.join(grid[row_idx])
        prefix = "THR>" if row_idx == thresh_row else "    "
        print(f"  {label} {prefix}|{line}|")

    t_start = f"{sampled_t[0]:.1f}s"
    t_end = f"{sampled_t[-1]:.1f}s"
    print(f"       {'':5}{t_start}{' ' * max(0, len(sampled_ear) - len(t_start) - len(t_end))}{t_end}")


def run():
    print("=" * 75)
    print("  LAB #10 — EAR Analysis: 50 Facial Landmarks → Drowsiness Detection")
    print("  Based on: Soukupová & Čech (2016), Wierwille et al. (1994)")
    print("=" * 75)

    fps = 60

    # --- Condition 1: Normal driving, no glasses ---
    print("\n" + "─" * 75)
    print("  CONDITION 1: Normal Attentive Driving (No Glasses)")
    print("─" * 75)
    normal_frames = generate_normal_driving(duration_s=10.0, fps=fps)
    normal_stats = analyze_sequence(normal_frames, "Normal (No Glasses)", fps)
    _print_stats(normal_stats)
    print("\n  EAR Time-Series (· = open, ● = closed):")
    print_text_chart(normal_stats["timestamps"], normal_stats["ear_avg_series"], EAR_CLOSED_THRESHOLD)

    # --- Condition 2: Drowsy sequence ---
    print("\n" + "─" * 75)
    print("  CONDITION 2: Progressive Drowsiness → Microsleep")
    print("─" * 75)
    drowsy_frames = generate_drowsy_sequence(duration_s=15.0, fps=fps)
    drowsy_stats = analyze_sequence(drowsy_frames, "Drowsy", fps)
    _print_stats(drowsy_stats)
    print("\n  EAR Time-Series (· = open, ● = closed):")
    print_text_chart(drowsy_stats["timestamps"], drowsy_stats["ear_avg_series"], EAR_CLOSED_THRESHOLD)

    # --- Condition 3: Glasses ---
    print("\n" + "─" * 75)
    print("  CONDITION 3: Normal Driving with Glasses (IR Reflection Artifacts)")
    print("─" * 75)
    glasses_frames = generate_glasses_sequence(duration_s=10.0, fps=fps)
    glasses_stats = analyze_sequence(glasses_frames, "Glasses", fps)
    _print_stats(glasses_stats)
    print("\n  EAR Time-Series (· = open, ● = closed):")
    print_text_chart(glasses_stats["timestamps"], glasses_stats["ear_avg_series"], EAR_CLOSED_THRESHOLD)

    # --- Comparison ---
    print("\n" + "=" * 75)
    print("  COMPARISON: No Glasses vs Glasses")
    print("=" * 75)
    print(f"  {'Metric':<30} {'No Glasses':>12} {'Glasses':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12}")
    print(f"  {'Mean EAR (L)':30} {normal_stats['mean_ear_l']:12.3f} {glasses_stats['mean_ear_l']:12.3f}")
    print(f"  {'Mean EAR (R)':30} {normal_stats['mean_ear_r']:12.3f} {glasses_stats['mean_ear_r']:12.3f}")
    print(f"  {'Std EAR (L)':30} {normal_stats['std_ear_l']:12.3f} {glasses_stats['std_ear_l']:12.3f}")
    print(f"  {'Std EAR (R)':30} {normal_stats['std_ear_r']:12.3f} {glasses_stats['std_ear_r']:12.3f}")
    print(f"  {'Real Blinks':30} {normal_stats['blinks_detected']:12d} {glasses_stats['blinks_detected']:12d}")
    print(f"  {'False Closures (glasses)':30} {normal_stats['false_closures']:12d} {glasses_stats['false_closures']:12d}")
    print(f"  {'PERCLOS':30} {normal_stats['perclos']:11.1%} {glasses_stats['perclos']:11.1%}")

    if glasses_stats["false_closure_frames"]:
        print(f"\n  ⚠  Glasses error frames: {glasses_stats['false_closure_frames']}")
        for fc in glasses_stats["false_closure_details"]:
            print(
                f"     Frame {fc['start_frame']}: IR reflection on left lens → "
                f"EAR dropped to {fc['min_ear']:.3f} for {fc['duration_ms']:.0f}ms (false closure)"
            )

    if glasses_stats["std_ear_l"] > normal_stats["std_ear_l"] * 1.3:
        ratio = glasses_stats["std_ear_l"] / normal_stats["std_ear_l"]
        print(f"\n  ⚠  Left eye EAR variance is {ratio:.1f}× higher with glasses (gọng kính noise)")

    print("\n" + "=" * 75)
    print("  [LAB #10 COMPLETE]")
    print("=" * 75)


def _print_stats(stats: dict):
    label = stats["label"]
    print(f"\n  Sequence: {label} ({stats['duration_s']:.1f}s @ {stats['total_frames']} frames)")
    print(f"  Mean EAR (L/R): {stats['mean_ear_l']:.3f} / {stats['mean_ear_r']:.3f}")
    print(f"  Std  EAR (L/R): {stats['std_ear_l']:.3f} / {stats['std_ear_r']:.3f}")
    print(f"  Blinks detected: {stats['blinks_detected']}")
    if stats["false_closures"] > 0:
        print(f"  False closures (glasses): {stats['false_closures']}")
    print(f"  PERCLOS: {stats['perclos']:.1%}")


if __name__ == "__main__":
    run()
