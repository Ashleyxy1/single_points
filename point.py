"""
手部关键点提取核心模块 (MediaPipe 0.10.x Tasks API)
"""
import numpy as np
from mediapipe.tasks.python import vision, BaseOptions

# 手部关键点名称 (21个)
LANDMARK_NAMES = [
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]


def create_hands_detector(model_path="hand_landmarker.task", num_hands=2,
                          confidence=0.5, running_mode="IMAGE"):
    """
    创建 MediaPipe HandLandmarker 检测器。
    running_mode: 'IMAGE' 用于静态图片, 'VIDEO' 用于视频流
    """
    mode = vision.RunningMode.IMAGE if running_mode == "IMAGE" else vision.RunningMode.VIDEO
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=num_hands,
        min_hand_detection_confidence=confidence,
        min_tracking_confidence=0.5,
        running_mode=mode,
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_raw_keypoints(detection_result):
    """
    从检测结果中提取原始关键点。
    返回: (126,) numpy 数组 [左手63维 + 右手63维]，未检测到的手用0填充。
    注意: 新API的坐标已被归一化到 [0,1]，z值以手腕为原点。
    """
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)

    if detection_result.hand_landmarks:
        for hand_lm, handedness in zip(detection_result.hand_landmarks,
                                        detection_result.handedness):
            kp = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm]).flatten()
            label = handedness[0].category_name
            if label == "Left":
                lh = kp
            elif label == "Right":
                rh = kp
    return np.concatenate([lh, rh])


def extract_normalized_keypoints(detection_result, image_width, image_height):
    """
    提取归一化关键点：以手腕为原点，用手掌大小做尺度归一化。
    返回: (126,) numpy 数组
    """
    keypoints = np.zeros(126)
    if detection_result.hand_landmarks:
        for hand_lm, handedness in zip(detection_result.hand_landmarks,
                                        detection_result.handedness):
            coords = np.array([[lm.x * image_width, lm.y * image_height,
                                lm.z * image_width] for lm in hand_lm])
            wrist = coords[0]
            coords -= wrist
            scale = np.linalg.norm(coords[9])  # 手腕到中指MCP距离
            if scale < 1e-6:
                scale = 1.0
            coords /= scale

            kp = coords.flatten()
            label = handedness[0].category_name
            if label == "Left":
                keypoints[0:63] = kp
            elif label == "Right":
                keypoints[63:126] = kp
    return keypoints


def count_raised_fingers(hand_landmarks, threshold=0.04):
    """
    统计伸出的手指数量。
    hand_landmarks: NormalizedLandmark 列表 (21个)
    """
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    raised = 0
    for tip, pip in zip(tips, pips):
        # 指尖 y 坐标小于 PIP 关节 y 坐标 → 手指伸直
        if hand_landmarks[tip].y < hand_landmarks[pip].y - threshold:
            raised += 1
    return raised


def draw_landmarks(image, detection_result):
    """在图像上绘制手部关键点和骨架。"""
    annotated = image.copy()
    if detection_result.hand_landmarks:
        for hand_lm in detection_result.hand_landmarks:
            vision.drawing_utils.draw_landmarks(
                annotated,
                hand_lm,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
            )
    return annotated
