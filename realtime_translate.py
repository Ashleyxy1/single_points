"""
实时手语翻译 (MediaPipe 0.10.x)
- 打开摄像头，实时检测手部关键点
- 使用训练好的分类器进行手势识别
- 显示预测结果和置信度
"""
import cv2
import numpy as np
import os
import sys
import json
import pickle
import time
import mediapipe as mp
from collections import deque, Counter

sys.path.insert(0, os.path.dirname(__file__))
from point import (create_hands_detector, extract_normalized_keypoints,
                   draw_landmarks, count_raised_fingers)


def load_model(model_dir="models"):
    """加载训练好的模型"""
    classifier_file = os.path.join(model_dir, "classifier.pkl")
    scaler_file = os.path.join(model_dir, "scaler.pkl")
    mapping_file = os.path.join(model_dir, "label_map.json")

    if not all(os.path.exists(f) for f in [classifier_file, scaler_file, mapping_file]):
        print("模型文件不完整，请先运行 train_model.py 训练模型")
        print(f"  需要: {classifier_file}, {scaler_file}, {mapping_file}")
        return None, None, None

    with open(classifier_file, "rb") as f:
        model = pickle.load(f)
    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)
    with open(mapping_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_label = {int(v): k for k, v in label_map.items()}
    print(f"模型加载成功: {len(label_map)} 个手势类别")
    return model, scaler, id_to_label


def get_prediction_with_confidence(model, scaler, keypoints):
    """获取预测结果和置信度"""
    from sklearn.neural_network import MLPClassifier

    X = scaler.transform([keypoints])

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
    elif hasattr(model, '_predict_proba_lr'):
        # KNeighborsClassifier
        proba = model.predict_proba(X)[0]
    else:
        pred = model.predict(X)[0]
        return int(pred), 1.0

    pred = int(np.argmax(proba))
    confidence = float(proba[pred])
    return pred, confidence


def main():
    model_dir = "models"
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]

    model, scaler, id_to_label = load_model(model_dir)
    if model is None:
        print("提示: 没有训练好的模型时，可以先运行查看关键点")
        print("运行 'python realtime_translate.py --keypoints' 进入关键点查看模式")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    detector = create_hands_detector(num_hands=2, confidence=0.5, running_mode="IMAGE")

    prediction_history = deque(maxlen=10)
    confidence_threshold = 0.6

    print("\n=== 实时手语翻译 ===")
    print("按 Q/ESC 退出")
    print("====================\n")

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

        if result.hand_landmarks:
            keypoints = extract_normalized_keypoints(result, w, h)

            if not np.allclose(keypoints[:63], 0) or not np.allclose(keypoints[63:], 0):
                pred_id, confidence = get_prediction_with_confidence(model, scaler, keypoints)
                prediction_history.append(pred_id)

                most_common = Counter(prediction_history).most_common(1)[0]
                smoothed_id, count = most_common
                stability = count / len(prediction_history)

                if confidence >= confidence_threshold and stability >= 0.5:
                    sign_name = id_to_label.get(smoothed_id, "?")
                else:
                    sign_name = "..."

                # 显示手指数量
                for i, hand_lm in enumerate(result.hand_landmarks):
                    fingers = count_raised_fingers(hand_lm)
                    cv2.putText(annotated, f"Hand{i+1} fingers: {fingers}",
                                (10, 95 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 200, 255), 2)

                # 预测结果
                color = (0, 255, 0) if confidence >= confidence_threshold else (0, 165, 255)
                cv2.putText(annotated, f"Sign: {sign_name}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(annotated, f"Conf: {confidence:.1%}  Stable: {stability:.1%}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                # Top-3 预测
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(scaler.transform([keypoints]))[0]
                    top3_idx = np.argsort(proba)[-3:][::-1]
                    for i, idx in enumerate(top3_idx):
                        name = id_to_label.get(idx, "?")
                        p = proba[idx]
                        cv2.putText(annotated, f"  {i+1}. {name}: {p:.1%}",
                                    (w - 220, 30 + i * 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        else:
            prediction_history.clear()

        cv2.imshow("Sign Language Translator", annotated)

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
