import warnings
from pathlib import Path

import cv2

from boulderbuddy.analysis.heuristics import analyze_frame, classify_movement
from boulderbuddy.config import VIDEO_OUTPUT_DIR
from boulderbuddy.pose.estimator import PoseEstimator
from boulderbuddy.pose.keypoints import extract_keypoints, get_hip_center
from boulderbuddy.video.loader import open_video
from boulderbuddy.video.writer import create_video_writer
from boulderbuddy.visualization.overlay import draw_feedback, draw_pose

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    input_video = Path("data/samples/test.mp4")
    output_video = VIDEO_OUTPUT_DIR / "annotated_test.mp4"

    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = open_video(input_video)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = create_video_writer(output_video, fps, width, height)
    estimator = PoseEstimator()

    frame_count = 0
    detected_frames = 0
    previous_hip = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_result = estimator.estimate(frame)

        if pose_result.detected:
            detected_frames += 1

        keypoints = extract_keypoints(pose_result, width, height)
        current_hip = get_hip_center(keypoints) if keypoints else None

        movement_state = classify_movement(current_hip, previous_hip)
        feedback = analyze_frame(
            movement_state=movement_state,
            pose_detected=pose_result.detected,
            keypoints=keypoints,
        )

        frame = draw_pose(frame, pose_result)
        frame = draw_feedback(frame, feedback)

        writer.write(frame)

        previous_hip = current_hip
        frame_count += 1

    cap.release()
    writer.release()
    estimator.close()

    print(f"Processed {frame_count} frames.")
    print(f"Pose detected in {detected_frames} frames.")
    print(f"Saved annotated video to: {output_video}")


if __name__ == "__main__":
    main()
