"""
手语数据采集工具 (MediaPipe 0.10.x)
- 所有操作都在 OpenCV 窗口中完成，不需要切换窗口
- 数字键 1-9 切换手势标签
- 左右箭头键 切换上一个/下一个标签
- SPACE 采集当前帧
- Q/ESC 保存并退出
"""
import os
import sys
import json
import signal
import numpy as np
import cv2
import mediapipe as mp

# 启动时屏蔽无关日志
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.dirname(__file__))
from point import (create_hands_detector, extract_normalized_keypoints,
                   draw_landmarks, count_raised_fingers)

_data_ref = {"keypoints": [], "labels": [], "label_map": {},
             "data_file": None, "labels_file": None, "mapping_file": None,
             "detector": None, "cap": None}


def _cleanup():
    if _data_ref.get("detector"):
        try:
            _data_ref["detector"].close()
        except Exception:
            pass
    if _data_ref.get("cap"):
        try:
            _data_ref["cap"].release()
        except Exception:
            pass
    cv2.destroyAllWindows()


def _save_data():
    if _data_ref["keypoints"]:
        np.save(_data_ref["data_file"],
                np.array(_data_ref["keypoints"], dtype=np.float32))
        np.save(_data_ref["labels_file"],
                np.array(_data_ref["labels"], dtype=np.int32))
        with open(_data_ref["mapping_file"], "w", encoding="utf-8") as f:
            json.dump(_data_ref["label_map"], f, ensure_ascii=False, indent=2)
        print(f"\n数据已保存: {len(_data_ref['keypoints'])} 个样本, "
              f"{len(_data_ref['label_map'])} 个类别")
    else:
        print("\n未采集任何数据，跳过保存")


def _signal_handler(sig, frame):
    print("\n检测到 Ctrl+C，正在保存...")
    _save_data()
    _cleanup()
    os._exit(0)


def _get_label_list(label_map):
    """返回有序的标签列表"""
    return sorted(label_map.keys(), key=lambda k: label_map[k])


def main():
    save_dir = "training_data"
    os.makedirs(save_dir, exist_ok=True)

    data_file = os.path.join(save_dir, "keypoints.npy")
    labels_file = os.path.join(save_dir, "labels.npy")
    mapping_file = os.path.join(save_dir, "label_map.json")

    if os.path.exists(data_file):
        all_keypoints = list(np.load(data_file))
        all_labels = list(np.load(labels_file))
        with open(mapping_file, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        print(f"已加载 {len(all_keypoints)} 个样本, {len(label_map)} 个类别")
    else:
        all_keypoints = []
        all_labels = []
        label_map = {}
        # 首次使用，添加默认标签
        print("首次使用，请先在终端中输入要采集的手势列表\n")
        print("示例: 输入 A B C D E (空格分隔)")
        raw = input("手势列表: ").strip()
        if raw:
            for name in raw.split():
                name = name.strip()
                if name and name not in label_map:
                    label_map[name] = len(label_map)
        if not label_map:
            # 预置一些常用标签
            for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                label_map[ch] = len(label_map)

    _data_ref["keypoints"] = all_keypoints
    _data_ref["labels"] = all_labels
    _data_ref["label_map"] = label_map
    _data_ref["data_file"] = data_file
    _data_ref["labels_file"] = labels_file
    _data_ref["mapping_file"] = mapping_file

    signal.signal(signal.SIGINT, _signal_handler)

    label_list = _get_label_list(label_map)
    current_idx = 0
    label_name = label_list[current_idx]
    label_id = label_map[label_name]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    detector = create_hands_detector(num_hands=2, confidence=0.5, running_mode="IMAGE")
    _data_ref["cap"] = cap
    _data_ref["detector"] = detector

    print(f"\n手势列表: {label_list}")
    print(f"当前手势: [{label_name}]")
    print("操作: 1-9=选标签 | ←→=切换标签 | SPACE=采集 | A=添加新标签 | Q=保存退出\n")

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

            # 左侧面板
            cv2.putText(annotated, f"Sign: [{label_name}]",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(annotated, f"Samples: {len(all_keypoints)}",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

            if result.hand_landmarks:
                for i, hand_lm in enumerate(result.hand_landmarks):
                    fingers = count_raised_fingers(hand_lm)
                    cv2.putText(annotated, f"Hand{i+1} fingers: {fingers}",
                                (10, 95 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 200, 255), 2)

            # 右侧面板：显示标签列表
            y_start = 30
            for i, name in enumerate(label_list):
                prefix = ">> " if i == current_idx else "   "
                color = (0, 255, 0) if i == current_idx else (180, 180, 180)
                text = f"{prefix}{i+1}. {name}"
                cv2.putText(annotated, text, (w - 200, y_start + i * 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 底部操作提示
            cv2.putText(annotated, "1-9:Label | <- ->:Switch | SPACE:Capture | A:Add | Q:Save&Quit",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Data Collection - Click this window first!", annotated)

            raw_key = cv2.waitKey(1)
            if raw_key == -1:
                pass
            elif raw_key == ord(" "):
                if result.hand_landmarks:
                    kp = extract_normalized_keypoints(result, w, h)
                    all_keypoints.append(kp)
                    all_labels.append(label_id)
                    print(f"  [+{len(all_keypoints)}] {label_name}")
                    # 每次采集后自动保存 (防止崩溃丢数据)
                    _save_data()
                else:
                    print("  [!] 未检测到手")

            elif raw_key == 27 or raw_key == ord("q") or raw_key == ord("Q"):
                break

            # 左右箭头键
            elif raw_key == 2424832 or raw_key == 81:
                current_idx = (current_idx - 1) % len(label_list)
                label_name = label_list[current_idx]
                label_id = label_map[label_name]
            elif raw_key == 2555904 or raw_key == 83:
                current_idx = (current_idx + 1) % len(label_list)
                label_name = label_list[current_idx]
                label_id = label_map[label_name]

            # 数字键 1-9: 快速切标签
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
    finally:
        _save_data()
        _cleanup()


if __name__ == "__main__":
    main()
