from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp


@dataclass
class PoseResult:
    landmarks: Any
    detected: bool


class PoseEstimator:
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate(self, frame) -> PoseResult:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        detected = results.pose_landmarks is not None
        return PoseResult(
            landmarks=results.pose_landmarks,
            detected=detected,
        )

    def close(self) -> None:
        self.pose.close()
