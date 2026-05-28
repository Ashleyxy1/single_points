"""
实时动态手势识别

工作流程:
  1. 摄像头实时捕获帧
  2. MotionSegmenter 监测手腕位移，自动分割手势起止
  3. 分割出的序列用 DTW-KNN 分类
  4. 显示识别结果、置信度、Top-3 预测

操作: 按 Q / ESC 退出
"""
import os
import sys
import json
import pickle
import time
import numpy as np
import cv2
import mediapipe as mp
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(__file__))
from point import create_hands_detector, draw_landmarks, count_raised_fingers
from dynamic_hand_dtw import (
    DTWKNN, MotionSegmenter, extract_raw_keypoints
)
from realtime_translate import put_chinese_texts
from ws_sender import create_sender

MODEL_DIR = "dynamic_data"
SEQUENCES_FILE = os.path.join(MODEL_DIR, "sequences.pkl")
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_map.json")


def load_model():
    """从 dynamic_data/ 加载训练好的 DTW-KNN 模型。"""
    if not os.path.exists(SEQUENCES_FILE):
        print(f"找不到 {SEQUENCES_FILE}，请先运行 collect_dynamic.py 采集数据")
        return None, None

    with open(SEQUENCES_FILE, "rb") as f:
        sequences = pickle.load(f)
    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_label = {int(v): k for k, v in label_map.items()}

    # 展开数据
    X = [seq for seq, _ in sequences]
    y = [label for _, label in sequences]

    dtw_knn = DTWKNN(n_neighbors=5, window=None, downsample=True, target_frames=30)
    dtw_knn.fit(X, y)

    print(f"模型加载成功: {len(sequences)} 个模板, {len(label_map)} 个类别")
    print(f"  类别: {list(label_map.keys())}")
    return dtw_knn, id_to_label


def main():
    dtw_knn, id_to_label = load_model()
    if dtw_knn is None:
        return

    ws_sender = None
    for arg in sys.argv[1:]:
        if arg.startswith("--ws="):
            ws_sender = create_sender(arg[5:])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    detector = create_hands_detector(num_hands=2, confidence=0.5, running_mode="IMAGE")
    segmenter = MotionSegmenter(
        buffer_seconds=3.0,
        fps=30,
        onset_threshold=0.015,
        offset_threshold=0.008,
        min_gesture_frames=10,
        max_gesture_frames=90,
        settle_frames=15,
    )

    # 预测历史（用于显示识别到的多个手势）
    result_history = deque(maxlen=5)
    last_recognition_time = 0
    current_result = None  # (gesture_name, confidence)
    last_printed = None

    print("\n=== 实时动态手势识别 ===")
    print("做动作即可，系统会自动检测并识别")
    print("按 Q / ESC 退出")
    print("========================\n")

    # 调参提示
    print("提示: 如果手势难以被检测到，可以调整阈值:")
    print(f"  onset_threshold={segmenter.onset_threshold} (越小越敏感)")
    print(f"  offset_threshold={segmenter.offset_threshold} (越小越难结束)")
    print(f"  settle_frames={segmenter.settle_frames} (越小判定结束越快)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = detector.detect(mp_image)

        annotated = draw_landmarks(frame, result)
        kp = extract_raw_keypoints(result)

        # ---- 运动分割 ----
        detected_seq = segmenter.update(kp)

        if detected_seq is not None:
            # 识别
            t0 = time.perf_counter()
            pred_id, confidence, passed = dtw_knn.predict_with_confidence(
                detected_seq, threshold=0.40
            )
            elapsed = (time.perf_counter() - t0) * 1000

            if passed:
                gesture_name = id_to_label.get(pred_id, "?")
                current_result = (gesture_name, confidence, len(detected_seq))
                result_history.append(current_result)
                last_recognition_time = time.time()

                if gesture_name != last_printed:
                    print(f"[识别] {gesture_name}  |  置信度: {confidence:.1%}  |  "
                          f"帧数: {len(detected_seq)}  |  DTW: {elapsed:.0f}ms")
                    if ws_sender:
                        ws_sender.send({
                            "gesture": gesture_name,
                            "confidence": round(confidence, 4),
                            "type": "dynamic",
                            "frames": len(detected_seq),
                            "dtw_ms": round(elapsed, 1),
                        })
                    last_printed = gesture_name
            else:
                last_printed = None

        # ---- 画面绘制 ----

        # 状态栏背景
        state = "RECORDING" if segmenter.is_gesturing else "IDLE"
        state_color = (0, 0, 255) if segmenter.is_gesturing else (0, 200, 0)
        cv2.putText(annotated, f"State: {state}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

        # 手指数量
        if result.hand_landmarks:
            for i, hand_lm in enumerate(result.hand_landmarks):
                fingers = count_raised_fingers(hand_lm)
                cv2.putText(annotated, f"Hand{i + 1} fingers: {fingers}",
                            (10, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 200, 255), 2)

        # Buffer 可视化
        buffer_size = len(segmenter.gesture_frames) if segmenter.is_gesturing else 0
        if buffer_size > 0:
            bar_width = int(buffer_size / segmenter.max_gesture_frames * 200)
            cv2.rectangle(annotated, (w - 220, 10), (w - 20, 30), (80, 80, 80), -1)
            cv2.rectangle(annotated, (w - 220, 10), (w - 220 + bar_width, 30),
                          (0, 200, 0), -1)
            cv2.putText(annotated, f"Buffer: {buffer_size}",
                        (w - 220, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 识别结果
        if current_result is not None:
            elapsed_since = time.time() - last_recognition_time
            if elapsed_since < 3.0:  # 显示最近3秒内的结果
                name, conf, seq_len = current_result
                alpha = max(0.3, 1.0 - elapsed_since / 3.0)  # 渐隐

                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, h - 200), (350, h), (30, 30, 30), -1)
                annotated = cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0)

                annotated = put_chinese_texts(annotated,
                    [(f"Gesture: {name}", 15, h - 150, 38, (0, 255, 0))])
                cv2.putText(annotated, f"Confidence: {conf:.1%}",
                            (15, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0), 2)
                cv2.putText(annotated, f"Frames: {seq_len}",
                            (15, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1)

        # 历史记录
        cv2.putText(annotated, "History:", (w - 220, h - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        for i, entry in enumerate(list(result_history)[-5:]):
            name, conf, _ = entry
            cv2.putText(annotated, f"  {name} ({conf:.0%})",
                        (w - 220, h - 95 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 底部提示
        cv2.putText(annotated, "Q/ESC: Quit  |  Perform gesture to recognize",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cv2.imshow("Dynamic Gesture Recognition", annotated)

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    if ws_sender:
        ws_sender.close()


if __name__ == "__main__":
    main()
