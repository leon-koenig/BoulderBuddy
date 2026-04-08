from __future__ import annotations

from typing import Dict, Optional

import mediapipe as mp

mp_pose = mp.solutions.pose


LANDMARK_NAMES = {
    "nose": mp_pose.PoseLandmark.NOSE,
    "left_shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": mp_pose.PoseLandmark.RIGHT_HIP,
    "left_knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "right_knee": mp_pose.PoseLandmark.RIGHT_KNEE,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
}


def extract_keypoints(
    pose_result,
    frame_width: int,
    frame_height: int,
    min_visibility: float = 0.3,
) -> Optional[Dict[str, dict]]:
    if not pose_result.detected or pose_result.landmarks is None:
        return None

    landmarks = pose_result.landmarks.landmark
    keypoints: Dict[str, dict] = {}

    for name, landmark_id in LANDMARK_NAMES.items():
        lm = landmarks[landmark_id.value]
        keypoints[name] = {
            "x": lm.x * frame_width,
            "y": lm.y * frame_height,
            "visibility": lm.visibility,
            "present": lm.visibility >= min_visibility,
        }

    return keypoints


def midpoint(a: dict, b: dict) -> dict:
    return {
        "x": (a["x"] + b["x"]) / 2.0,
        "y": (a["y"] + b["y"]) / 2.0,
        "visibility": min(a.get("visibility", 1.0), b.get("visibility", 1.0)),
        "present": a.get("present", True) and b.get("present", True),
    }


def get_hip_center(keypoints: Dict[str, dict]) -> Optional[dict]:
    left_hip = keypoints.get("left_hip")
    right_hip = keypoints.get("right_hip")

    if left_hip is None or right_hip is None:
        return None

    if not left_hip.get("present", False) or not right_hip.get("present", False):
        return None

    return midpoint(left_hip, right_hip)


def get_shoulder_center(keypoints: Dict[str, dict]) -> Optional[dict]:
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")

    if left_shoulder is None or right_shoulder is None:
        return None

    if not left_shoulder.get("present", False) or not right_shoulder.get(
        "present", False
    ):
        return None

    return midpoint(left_shoulder, right_shoulder)
