def analyze_pose(pose_result) -> list[str]:
    feedback = []

    if not pose_result.detected:
        feedback.append("No climber pose detected in frame.")

    return feedback
