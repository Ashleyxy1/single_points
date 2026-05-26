"""
动态手势数据采集工具

操作说明:
  - 数字键 1-9: 切换手势标签
  - 左右箭头: 翻页切换标签
  - A: 添加新手势标签
  - SPACE: 开始/停止录制（按下开始，再按停止并保存）
  - Q / ESC: 保存退出

录制期间画面会显示红色 "REC" 指示器和帧计数。
"""
import os
import sys
import json
import pickle
import signal
import numpy as np
import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(__file__))
from point import create_hands_detector, draw_landmarks
from dynamic_hand_dtw import extract_raw_keypoints

SAVE_DIR = "dynamic_data"
SEQUENCES_FILE = os.path.join(SAVE_DIR, "sequences.pkl")
LABEL_MAP_FILE = os.path.join(SAVE_DIR, "label_map.json")

# 全局引用，用于信号处理和清理
_ref = {}


def _cleanup():
    if _ref.get("detector"):
        try:
            _ref["detector"].close()
        except Exception:
            pass
    if _ref.get("cap"):
        try:
            _ref["cap"].release()
        except Exception:
            pass
    cv2.destroyAllWindows()


def _save_all(sequences, label_map):
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(SEQUENCES_FILE, "wb") as f:
        pickle.dump(sequences, f)
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)


def _signal_handler(sig, frame):
    print("\n检测到 Ctrl+C，正在保存...")
    _save_all(_ref.get("sequences", []), _ref.get("label_map", {}))
    _cleanup()
    os._exit(0)


def _get_label_list(label_map):
    return sorted(label_map.keys(), key=lambda k: label_map[k])


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 加载已有数据
    if os.path.exists(SEQUENCES_FILE) and os.path.exists(LABEL_MAP_FILE):
        with open(SEQUENCES_FILE, "rb") as f:
            sequences = pickle.load(f)
        with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        print(f"已加载 {len(sequences)} 个动态手势序列, {len(label_map)} 个类别")
    else:
        sequences = []
        label_map = {}
        print("首次使用，请先在终端中输入要采集的动态手势列表\n")
        print("示例: wave 画圈 滑动 剪刀 (空格分隔)")
        raw = input("手势列表: ").strip()
        if raw:
            for name in raw.split():
                name = name.strip()
                if name and name not in label_map:
                    label_map[name] = len(label_map)
        if not label_map:
            label_map["wave"] = 0
            label_map["circle"] = 1

    _ref["sequences"] = sequences
    _ref["label_map"] = label_map

    signal.signal(signal.SIGINT, _signal_handler)

    label_list = _get_label_list(label_map)
    current_idx = 0
    label_name = label_list[current_idx]
    label_id = label_map[label_name]

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    _ref["cap"] = cap

    detector = create_hands_detector(num_hands=2, confidence=0.5, running_mode="IMAGE")
    _ref["detector"] = detector

    # 录制状态
    is_recording = False
    recording_buffer = []
    prev_key = -1

    print(f"\n动态手势列表: {label_list}")
    print(f"当前手势: [{label_name}]")
    print("操作: 1-9=选标签 | ←→=切换 | SPACE=录制 | A=添加 | Q=保存退出\n")
    print("提示: 按 SPACE 开始录制 → 做动作 → 再按 SPACE 停止保存")

    try:
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

            # ---- 提取关键点 ----
            kp = extract_raw_keypoints(result)

            # ---- 录制逻辑 ----
            if is_recording:
                recording_buffer.append(kp.copy())
                # 录制指示器
                overlay = annotated.copy()
                cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 200), -1)
                annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
                cv2.putText(annotated, "REC", (15, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
                cv2.putText(annotated, f"Frames: {len(recording_buffer)}",
                            (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                # 闪烁提示
                if (len(recording_buffer) // 15) % 2 == 0:
                    cv2.circle(annotated, (w - 30, 40), 12, (0, 0, 255), -1)

            # ---- 左侧信息面板 ----
            status_color = (0, 0, 255) if is_recording else (0, 255, 0)
            cv2.putText(annotated, f"Sign: [{label_name}]",
                        (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
            sample_count = sum(1 for _, lbl in sequences if lbl == label_id)
            cv2.putText(annotated, f"Samples: {sample_count}  |  Total: {len(sequences)}",
                        (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # ---- 右侧标签列表 ----
            y_start = 30
            for i, name in enumerate(label_list):
                prefix = ">> " if i == current_idx else "   "
                color = (0, 255, 0) if i == current_idx else (180, 180, 180)
                text = f"{prefix}{i + 1}. {name}"
                cv2.putText(annotated, text, (w - 210, y_start + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # ---- 底部提示 ----
            cv2.putText(annotated, "1-9:Label | <- ->:Switch | SPACE:Record | A:Add | Q:Save&Quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Dynamic Gesture Collection", annotated)

            # ---- 键盘处理 ----
            raw_key = cv2.waitKey(1)

            if raw_key == ord(" "):
                if prev_key != ord(" "):   # 只响应按键按下（非长按）
                    if is_recording:
                        # 停止录制并保存
                        if len(recording_buffer) >= 5:
                            seq_arr = np.array(recording_buffer, dtype=np.float32)
                            sequences.append((seq_arr, label_id))
                            _save_all(sequences, label_map)
                            print(f"  [+] {label_name}: {len(recording_buffer)} 帧 -> "
                                  f"总计 {len(sequences)} 个序列")
                        else:
                            print(f"  [!] 序列太短 ({len(recording_buffer)} 帧)，已丢弃")
                        recording_buffer = []
                    else:
                        # 开始录制
                        recording_buffer = []
                        print(f"  [REC] 录制 '{label_name}' ...")
                    is_recording = not is_recording

            elif raw_key == 27 or raw_key == ord("q") or raw_key == ord("Q"):
                if is_recording:
                    print("  录制中，请先按 SPACE 停止录制")
                else:
                    break

            elif raw_key == 81:   # 左箭头
                current_idx = (current_idx - 1) % len(label_list)
                label_name = label_list[current_idx]
                label_id = label_map[label_name]
            elif raw_key == 83:   # 右箭头
                current_idx = (current_idx + 1) % len(label_list)
                label_name = label_list[current_idx]
                label_id = label_map[label_name]

            elif ord("1") <= raw_key <= ord("9"):
                idx = raw_key - ord("1")
                if idx < len(label_list):
                    current_idx = idx
                    label_name = label_list[current_idx]
                    label_id = label_map[label_name]

            elif raw_key == ord("a") or raw_key == ord("A"):
                print("在终端输入新手势名称: ", end="")
                new_name = input().strip()
                if new_name:
                    if new_name not in label_map:
                        label_map[new_name] = len(label_map)
                    label_list = _get_label_list(label_map)
                    current_idx = label_list.index(new_name)
                    label_name = new_name
                    label_id = label_map[label_name]
                    print(f"  已添加并切换到: [{label_name}]")

            prev_key = raw_key

    finally:
        _save_all(sequences, label_map)
        _cleanup()
        print(f"\n数据已保存到 {SAVE_DIR}/")
        print(f"  序列: {len(sequences)} 个")
        print(f"  类别: {len(label_map)} 个")


if __name__ == "__main__":
    main()
