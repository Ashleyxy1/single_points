# 手语识别系统

基于 MediaPipe 手部关键点 + 机器学习的实时手语/手势识别。

## 项目结构

```
model_points/
├── point.py                  # 公共核心模块
├── hand_landmarker.task      # MediaPipe 手部检测模型（~7.8MB）
├── flowchart.md              # 系统流程图（Mermaid）
│
├── camear_png.py             # 简单拍照工具
├── collect_data.py           # 静态手势数据采集
├── train_model.py            # 静态手势模型训练（MLP）
├── realtime_translate.py     # 静态手势实时识别
│
├── dynamic_hand_dtw.py       # 动态手势核心模块（DTW+KNN）
├── collect_dynamic.py        # 动态手势数据采集
├── realtime_dynamic.py       # 动态手势实时识别
│
├── ws_sender.py              # WebSocket 发送模块
├── ws_receiver.py            # WebSocket 网关（接收端参考实现）
│
├── training_data/            # 静态手势训练数据
│   ├── keypoints.npy
│   ├── labels.npy
│   └── label_map.json
│
├── models/                   # 静态手势模型
│   ├── classifier.pkl
│   ├── scaler.pkl
│   └── label_map.json
│
└── dynamic_data/             # 动态手势数据
    ├── sequences.pkl
    └── label_map.json
```

---

## 文件说明

### 公共模块与资源

#### `point.py` — 公共核心模块

所有脚本共享的基础功能：

- `create_hands_detector()` — 创建 MediaPipe HandLandmarker 检测器
- `extract_raw_keypoints()` — 提取 126 维原始关键点（21 个关键点 x 3 坐标 x 2 只手），保留手腕绝对位置
- `extract_normalized_keypoints()` — 提取归一化关键点，手腕居中 + 手掌尺度归一化
- `count_raised_fingers()` — 统计伸出的手指数量
- `draw_landmarks()` — 在图像上绘制手部骨架

#### `hand_landmarker.task`

MediaPipe 官方手部关键点检测模型文件。无需自己训练。

#### `flowchart.md`

用 Mermaid 绘制的系统流程图，展示从数据采集到模型训练再到实时推理的完整流水线。

---

### 静态手势识别（MLP 方案）

适用于**单帧手型**识别，如字母手势、数字手势等。

#### `collect_data.py` — 静态手势数据采集

打开摄像头，手动为每一帧打标签并采集：

| 操作 | 功能 |
|---|---|
| 1-9 | 切换到对应标签 |
| <- -> | 上/下一个标签 |
| SPACE | 采集当前帧（保存关键点 + 标签） |
| A | 添加新手势标签 |
| Q / ESC | 保存并退出 |

数据存入 `training_data/`。

#### `train_model.py` — 静态手势模型训练

加载 `training_data/` 中的 `.npy` 文件，训练 MLP 神经网络分类器：

- 特征标准化（StandardScaler）
- 训练集/测试集 8:2 分层划分
- MLP（256 -> 128 -> 64，ReLU，Adam）
- Early Stopping + 交叉验证
- 分类报告评估

模型保存到 `models/`。

#### `realtime_translate.py` — 静态手势实时识别

打开摄像头，逐帧预测手势：

- PIL 渲染中文文字（解决 OpenCV `putText` 不支持中文的问题）
- **置信度加权投票**：每帧票重 = 置信度，高置信度帧影响力更大
- **迟滞 (Hysteresis)**：锁定后需连续 4 帧不一致才切换，防止抖动
- 终端打印识别结果（变化时打印，不刷屏）
- 可选 WebSocket 发送（`--ws=ws://地址:端口`）

---

### 动态手势识别（DTW+KNN 方案）

适用于**有时序信息**的手势，如挥手、画圈、滑动等。

核心思路：不只看「此刻的手型」，而是看「一段时间内手的运动轨迹」。用 **DTW (Dynamic Time Warping)** 对齐不同快慢的序列，再用 KNN 分类。**不需要训练，采集完直接可用。**

#### `dynamic_hand_dtw.py` — 动态手势核心模块

- **关键点提取**：保留手腕绝对位置（运动轨迹需要手腕坐标）
- **序列预处理**：指数平滑去噪 -> Z-score 归一化 -> 下采样到固定帧数
- **DTW 算法**：`dtw_distance()` 含 Sakoe-Chiba band 约束，`dtw_align()` 含回溯路径
- **分类器**：`DTWKNN` 类，用 DTW 距离替代欧氏距离的 KNN
- **运动分割**：`MotionSegmenter` 类，通过手腕位移自动检测手势起止

#### `collect_dynamic.py` — 动态手势数据采集

采集**完整动作序列**而非单帧：

| 操作 | 功能 |
|---|---|
| 1-9 | 切换到对应标签 |
| <- -> | 上/下一个标签 |
| SPACE | 按一下开始录制 -> 做动作 -> 再按一下停止并保存 |
| A | 添加新手势标签 |
| Q / ESC | 保存并退出 |

录制期间画面显示红色 "REC" 和帧计数。数据存入 `dynamic_data/sequences.pkl`。

#### `realtime_dynamic.py` — 实时动态手势识别

打开摄像头，自动检测和识别动态手势：

1. `MotionSegmenter` 监测手腕位移，自动分割手势起止
2. 分割出的序列送入 `DTWKNN` 分类
3. 终端打印识别结果（变化时打印）
4. 可选 WebSocket 发送（`--ws=ws://地址:端口`）

---

### WebSocket 通信

将识别结果实时发送到网关。

#### `ws_sender.py` — 发送模块（库）

被识别脚本 `import`，不需要单独运行。提供后台线程发送、自动重连、消息队列。

#### `ws_receiver.py` — 网关接收端（参考实现）

```bash
pip install websockets websocket-client
python ws_receiver.py              # 默认监听 0.0.0.0:8765
python ws_receiver.py --port 9000  # 自定义端口
```

发送的消息格式：

```json
// 静态手势
{"gesture": "你好", "confidence": 0.853, "stability": 0.875, "type": "static", "timestamp": 1779893167.4}

// 动态手势
{"gesture": "hello", "confidence": 0.824, "type": "dynamic", "frames": 28, "dtw_ms": 42.3, "timestamp": 1779893167.4}
```

---

## 快速开始

### 静态手势

```bash
python collect_data.py                              # 1. 采集数据
python train_model.py                                # 2. 训练模型
python realtime_translate.py                         # 3. 实时识别（本地）
python realtime_translate.py --ws=ws://localhost:8765 # 3. 实时识别 + 发送到网关
```

### 动态手势

```bash
python collect_dynamic.py                            # 1. 采集数据（录制动作序列）
python realtime_dynamic.py                           # 2. 实时识别（本地）
python realtime_dynamic.py --ws=ws://localhost:8765   # 2. 实时识别 + 发送到网关
```

### 配合网关

```bash
# 终端 1: 启动网关
python ws_receiver.py

# 终端 2: 启动识别（任选一个）
python realtime_translate.py --ws=ws://localhost:8765
python realtime_dynamic.py --ws=ws://localhost:8765
```
