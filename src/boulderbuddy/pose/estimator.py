from dataclasses import dataclass
from typing import Any


@dataclass
class PoseResult:
    landmarks: Any
    detected: bool


class PoseEstimator:
    def __init__(self) -> None:
        pass

    def estimate(self, frame) -> PoseResult:
        return PoseResult(landmarks=None, detected=False)
