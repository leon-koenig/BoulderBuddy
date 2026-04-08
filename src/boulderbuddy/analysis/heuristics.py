from __future__ import annotations

import math
from typing import Optional


def angle(a: dict, b: dict, c: dict) -> float:
    ab = (a["x"] - b["x"], a["y"] - b["y"])
    cb = (c["x"] - b["x"], c["y"] - b["y"])

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.hypot(*ab)
    mag_cb = math.hypot(*cb)

    if mag_ab == 0 or mag_cb == 0:
        return 0.0

    cos_angle = dot / (mag_ab * mag_cb)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def elbow_angle(keypoints: dict, side: str = "left") -> float:
    shoulder = keypoints[f"{side}_shoulder"]
    elbow = keypoints[f"{side}_elbow"]
    wrist = keypoints[f"{side}_wrist"]

    return angle(shoulder, elbow, wrist)


def classify_movement(
    current_hip: Optional[dict],
    previous_hip: Optional[dict],
    threshold_px: float = 8.0,
) -> str:
    if current_hip is None or previous_hip is None:
        return "unknown"

    dx = current_hip["x"] - previous_hip["x"]
    dy = current_hip["y"] - previous_hip["y"]
    distance = math.hypot(dx, dy)

    if distance >= threshold_px:
        return "moving"

    return "stable"


def analyze_frame(
    movement_state: str,
    pose_detected: bool,
    keypoints: Optional[dict] = None,
) -> list[str]:
    if not pose_detected or keypoints is None:
        return ["No climber pose detected"]

    feedback: list[str] = []

    left_elbow = elbow_angle(keypoints, "left")
    right_elbow = elbow_angle(keypoints, "right")
    avg_elbow = (left_elbow + right_elbow) / 2.0

    if avg_elbow < 70:
        feedback.append("Arms heavily bent")

    if movement_state == "moving":
        feedback.append("Movement detected")
    elif movement_state == "stable":
        feedback.append("Stable position")
    else:
        feedback.append("Pose detected")

    return feedback
