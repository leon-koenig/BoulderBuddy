import cv2


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
