from pathlib import Path
import cv2

from boulderbuddy.config import VIDEO_OUTPUT_DIR
from boulderbuddy.video.loader import open_video
from boulderbuddy.video.writer import create_video_writer
from boulderbuddy.pose.estimator import PoseEstimator
from boulderbuddy.analysis.heuristics import analyze_pose
from boulderbuddy.visualization.overlay import draw_feedback


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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_result = estimator.estimate(frame)
        feedback = analyze_pose(pose_result)
        frame = draw_feedback(frame, feedback)

        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()

    print(f"Processed {frame_count} frames.")
    print(f"Saved annotated video to: {output_video}")


if __name__ == "__main__":
    main()
