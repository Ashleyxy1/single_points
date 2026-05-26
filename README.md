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
├── train_model.py            # 静态手势模型训练（MLP/KNN）
├── realtime_translate.py     # 静态手势实时识别
│
├── dynamic_hand_dtw.py       # 动态手势核心模块（DTW+KNN）
├── collect_dynamic.py        # 动态手势数据采集
├── realtime_dynamic.py       # 动态手势实时识别
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
- `extract_raw_keypoints()` — 提取 126 维原始关键点（21 个关键点 × 3 坐标 × 2 只手），保留手腕绝对位置
- `extract_normalized_keypoints()` — 提取归一化关键点，手腕居中 + 手掌尺度归一化
- `count_raised_fingers()` — 统计伸出的手指数量
- `draw_landmarks()` — 在图像上绘制手部骨架

#### `hand_landmarker.task`

MediaPipe 官方手部关键点检测模型文件。无需自己训练。

#### `flowchart.md`

用 Mermaid 绘制的系统流程图，展示从数据采集到模型训练再到实时推理的完整流水线。

---

### 静态手势识别（原有方案）

适用于**单帧手型**识别，如 A-Z 字母手势、数字手势等。

#### `collect_data.py` — 静态手势数据采集

打开摄像头，手动为每一帧打标签并采集：

| 操作 | 功能 |
|---|---|
| 1-9 | 切换到对应标签 |
| ← → | 上/下一个标签 |
| SPACE | 采集当前帧（保存关键点 + 标签） |
| A | 添加新手势标签 |
| Q / ESC | 保存并退出 |

数据存入 `training_data/`。

#### `train_model.py` — 静态手势模型训练

加载 `training_data/` 中的 `.npy` 文件，训练分类器：

- 特征标准化（StandardScaler）
- 训练集/测试集 8:2 分层划分
- MLP 神经网络（256 → 128 → 64，ReLU，Adam）
- 可选 KNN（`KNeighborsClassifier`）
- 交叉验证 + 分类报告评估

模型保存到 `models/`。

#### `realtime_translate.py` — 静态手势实时识别

打开摄像头，逐帧预测手势：

- 时序平滑（最近 10 帧多数投票）
- 双重过滤（置信度 ≥ 60% + 稳定性 ≥ 50%）
- 显示预测结果、置信度、Top-3 候选

#### `camear_png.py` — 拍照工具

简单的 OpenCV 拍照脚本，按 SPACE 保存图片到 `hand_sign_data/`。独立于手势识别流水线。

---

### 动态手势识别（DTW+KNN 方案）

适用于**有时序信息**的手势，如挥手、画圈、滑动等。

核心思路：不只看「此刻的手型」，而是看「一段时间内手的运动轨迹」。用 **DTW (Dynamic Time Warping)** 对齐不同快慢的序列，再用 KNN 分类。

#### `dynamic_hand_dtw.py` — 动态手势核心模块

- **关键点提取**：`extract_raw_keypoints()` — 保留手腕绝对位置（运动轨迹需要手腕坐标）
- **序列预处理**：
  - `smooth_sequence()` — 指数平滑去噪
  - `normalize_sequence()` — Z-score 归一化（消除位置偏移，保留轨迹形状）
  - `downsample_sequence()` — 线性插值到固定帧数
- **DTW 算法**：
  - `dtw_distance()` — 含 Sakoe-Chiba band 约束的 DTW 距离
  - `dtw_align()` — 含回溯路径的 DTW 对齐
- **分类器**：`DTWKNN` 类 — 用 DTW 距离替代欧氏距离的 KNN
- **运动分割**：`MotionSegmenter` 类 — 通过手腕位移自动检测手势起止

#### `collect_dynamic.py` — 动态手势数据采集

与静态采集不同，这里采集的是**完整动作序列**而非单帧：

| 操作 | 功能 |
|---|---|
| 1-9 | 切换到对应标签 |
| ← → | 上/下一个标签 |
| SPACE | 按一下开始录制 → 做动作 → 再按一下停止并保存 |
| A | 添加新手势标签 |
| Q / ESC | 保存并退出 |

录制期间画面显示红色 "REC" 和帧计数。数据存入 `dynamic_data/sequences.pkl`。

#### `realtime_dynamic.py` — 实时动态手势识别

打开摄像头，自动检测和识别动态手势：

1. `MotionSegmenter` 监测手腕位移，自动分割手势起止
2. 分割出的序列送入 `DTWKNN` 分类
3. 显示识别结果、置信度、历史记录

如果手势识别不够灵敏，可以调整 `MotionSegmenter` 的阈值参数：

- `onset_threshold` — 越小越容易触发录制（更敏感）
- `offset_threshold` — 越小越难结束录制
- `settle_frames` — 越小判定结束越快

---

## 快速开始

### 静态手势

```bash
python collect_data.py      # 1. 采集数据
python train_model.py        # 2. 训练模型
python realtime_translate.py # 3. 实时识别
```

### 动态手势

```bash
python collect_dynamic.py    # 1. 采集数据（录制动作序列）
python realtime_dynamic.py   # 2. 实时识别（自动分割+分类）
```

动态方案不需要单独的「训练」步骤——DTW-KNN 在加载时直接存储所有模板，识别时实时计算 DTW 距离。
