"""
实时手语翻译 (MediaPipe 0.10.x)
- 打开摄像头，实时检测手部关键点
- 使用训练好的分类器进行手势识别
- 显示预测结果和置信度
- 结果同时打印到终端
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
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from point import (create_hands_detector, extract_normalized_keypoints,
                   draw_landmarks, count_raised_fingers)
from ws_sender import create_sender


# ---- 字体缓存（只加载一次）----
_fonts = {}

def _get_font(size):
    if size not in _fonts:
        for font_path in [
            "C:/Windows/Fonts/msyh.ttf",
            "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/simsun.ttc",
        ]:
            try:
                _fonts[size] = ImageFont.truetype(font_path, size)
                break
            except (IOError, OSError):
                continue
        if size not in _fonts:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def put_chinese_texts(img, texts):
    """批量绘制中文到图像。texts: [(text, x, y, font_size, (b,g,r)), ...]
    每帧只做一次 RGB↔BGR 转换，比逐个调用快 3-4 倍。"""
    if not texts:
        return img
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    for text, x, y, size, color in texts:
        b, g, r = int(color[0]), int(color[1]), int(color[2])
        draw.text((x, y), text, font=_get_font(size), fill=(r, g, b))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def load_model(model_dir="models"):
    """加载训练好的模型"""
    classifier_file = os.path.join(model_dir, "classifier.pkl")
    scaler_file = os.path.join(model_dir, "scaler.pkl")
    mapping_file = os.path.join(model_dir, "label_map.json")

    if not all(os.path.exists(f) for f in [classifier_file, scaler_file, mapping_file]):
        print("模型文件不完整，请先运行 train_model.py 训练模型")
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
    X = scaler.transform([keypoints])
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]
    elif hasattr(model, '_predict_proba_lr'):
        proba = model.predict_proba(X)[0]
    else:
        pred = model.predict(X)[0]
        return int(pred), 1.0
    pred = int(np.argmax(proba))
    confidence = float(proba[pred])
    return pred, confidence


def main():
    model_dir = "models"
    ws_sender = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--ws="):
                ws_sender = create_sender(arg[5:])
            else:
                model_dir = arg

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

    prediction_history = deque(maxlen=8)
    confidence_threshold = 0.55
    last_printed = None
    locked_gesture = None      # 当前锁定的手势（迟滞用）
    lock_counter = 0           # 锁定状态下连续不匹配计数

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

        # 收集需要渲染的中文文本
        chinese_texts = []

        if result.hand_landmarks:
            keypoints = extract_normalized_keypoints(result, w, h)

            if not np.allclose(keypoints[:63], 0) or not np.allclose(keypoints[63:], 0):
                pred_id, confidence = get_prediction_with_confidence(model, scaler, keypoints)
                prediction_history.append((pred_id, confidence))

                # 置信度加权投票：每帧的票重 = 置信度
                weighted_votes = {}
                for pid, conf in prediction_history:
                    weighted_votes[pid] = weighted_votes.get(pid, 0) + conf
                total_weight = sum(weighted_votes.values())
                sorted_votes = sorted(weighted_votes.items(), key=lambda x: x[1], reverse=True)
                smoothed_id, winner_weight = sorted_votes[0]
                stability = winner_weight / total_weight if total_weight > 0 else 0

                if confidence >= confidence_threshold and stability >= 0.55:
                    candidate_name = id_to_label.get(smoothed_id, "?")
                    # 迟滞：如果已锁定了某个手势，需要更多证据才切换
                    if locked_gesture is None:
                        sign_name = candidate_name
                        locked_gesture = candidate_name
                        lock_counter = 0
                    elif candidate_name == locked_gesture:
                        lock_counter = 0
                        sign_name = candidate_name
                    else:
                        lock_counter += 1
                        if lock_counter >= 4:  # 连续4帧不一致才切换
                            sign_name = candidate_name
                            locked_gesture = candidate_name
                            lock_counter = 0
                        else:
                            sign_name = locked_gesture  # 保持旧结果
                else:
                    sign_name = "..."
                    if lock_counter >= 6:  # 不确定持续太久，释放锁定
                        locked_gesture = None
                        lock_counter = 0

                # 终端打印 + WebSocket 发送
                if sign_name not in ("...", "?") and sign_name != last_printed:
                    print(f"[识别] {sign_name}  |  置信度: {confidence:.1%}  |  稳定性: {stability:.1%}")
                    if ws_sender:
                        ws_sender.send({
                            "gesture": sign_name,
                            "confidence": round(confidence, 4),
                            "stability": round(stability, 4),
                            "type": "static",
                        })
                    last_printed = sign_name

                # 手指数量
                for i, hand_lm in enumerate(result.hand_landmarks):
                    fingers = count_raised_fingers(hand_lm)
                    cv2.putText(annotated, f"Hand{i+1} fingers: {fingers}",
                                (10, 115 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 200, 255), 2)

                # 预测结果 + 置信度
                color = (0, 255, 0) if confidence >= confidence_threshold else (0, 165, 255)
                chinese_texts.append((f"Sign: {sign_name}", 10, 15, 36, color))
                cv2.putText(annotated, f"Conf: {confidence:.1%}  Stable: {stability:.1%}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                # Top-3
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(scaler.transform([keypoints]))[0]
                    top3_idx = np.argsort(proba)[-3:][::-1]
                    for i, idx in enumerate(top3_idx):
                        name = id_to_label.get(idx, "?")
                        p = proba[idx]
                        chinese_texts.append(
                            (f"  {i+1}. {name}: {p:.1%}", w - 220, 10 + i * 28, 20, (200, 200, 200)))
        else:
            prediction_history.clear()

        # 批量渲染中文（每帧只做一次 PIL 转换）
        annotated = put_chinese_texts(annotated, chinese_texts)

        cv2.imshow("Sign Language Translator", annotated)

        if cv2.waitKey(1) & 0xFF in (27, ord("q")):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()
    if ws_sender:
        ws_sender.close()


if __name__ == "__main__":
    main()
