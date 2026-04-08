from pathlib import Path
import cv2


def create_video_writer(output_path: str | Path, fps: float, width: int, height: int):
    output_path = str(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for: {output_path}")
    return writer
