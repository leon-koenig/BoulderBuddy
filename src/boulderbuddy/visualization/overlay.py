import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def draw_pose(frame, pose_result):
    if pose_result.detected and pose_result.landmarks is not None:
        mp_drawing.draw_landmarks(
            frame,
            pose_result.landmarks,
            mp_pose.POSE_CONNECTIONS,
        )
    return frame


def draw_feedback(frame, feedback: list[str]):
    y = 30
    for line in feedback[:3]:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30
    return frame
