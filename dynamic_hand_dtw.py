"""
DTW + KNN 动态手势识别核心模块

- DTW (Dynamic Time Warping) 对齐不同长度的关键点序列
- KNN 用 DTW 距离替代欧氏距离进行分类
- 支持序列平滑、归一化、运动检测
"""
import numpy as np
from collections import Counter, deque

# ----------------------------------------------------------------
# 关键点提取（保留手腕绝对位置，用于运动轨迹捕获）
# ----------------------------------------------------------------

def extract_raw_keypoints(detection_result):
    """从 MediaPipe 检测结果提取原始关键点 (126,)。
    保留 MediaPipe 归一化坐标 [0,1]，不做手腕居中。
    左手63维 + 右手63维，未检测到的手用0填充。"""
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


def get_active_hand_mask(sequence):
    """返回活跃手掩码。sequence 形状 (T, 126)。
    检测左右手是否有运动，返回 (use_left: bool, use_right: bool)。"""
    lh_motion = np.std(sequence[:, 0:3], axis=0).sum()   # 左手腕位移
    rh_motion = np.std(sequence[:, 63:66], axis=0).sum()  # 右手腕位移
    threshold = 0.005
    return lh_motion > threshold, rh_motion > threshold


# ----------------------------------------------------------------
# 序列预处理
# ----------------------------------------------------------------

def smooth_sequence(sequence, alpha=0.4):
    """指数平滑去噪。sequence 形状 (T, D)。"""
    smoothed = sequence.copy()
    for t in range(1, len(sequence)):
        smoothed[t] = alpha * sequence[t] + (1 - alpha) * smoothed[t - 1]
    return smoothed


def normalize_sequence(sequence):
    """Z-score 归一化：每个维度减去均值除以标准差。
    消除绝对位置偏移，只保留运动轨迹的形状。"""
    mean = sequence.mean(axis=0, keepdims=True)
    std = sequence.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (sequence - mean) / std


def downsample_sequence(sequence, target_frames=30):
    """均匀下采样/上采样到固定帧数（线性插值）。"""
    n = len(sequence)
    if n < 3:
        return sequence
    indices = np.linspace(0, n - 1, target_frames)
    result = np.zeros((target_frames, sequence.shape[1]))
    for i, idx in enumerate(indices):
        lo, hi = int(np.floor(idx)), int(np.ceil(idx))
        if lo == hi:
            result[i] = sequence[lo]
        else:
            frac = idx - lo
            result[i] = (1 - frac) * sequence[lo] + frac * sequence[hi]
    return result


# ----------------------------------------------------------------
# DTW 距离
# ----------------------------------------------------------------

def dtw_distance(seq1, seq2, window=None):
    """计算两个序列的 DTW 距离（含 Sakoe-Chiba window 约束）。

    参数:
        seq1: (N, D) 数组
        seq2: (M, D) 数组
        window: 窗口宽度，None 则自动取 max(N, M) // 4

    返回: DTW 累积距离（标量）
    """
    n, m = len(seq1), len(seq2)
    if window is None:
        window = max(n, m) // 4
    window = max(window, abs(n - m))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j],      # 插入
                                    dtw[i, j - 1],      # 删除
                                    dtw[i - 1, j - 1])  # 匹配

    return dtw[n, m]


def dtw_align(seq1, seq2, window=None):
    """DTW 对齐并返回对齐后的两条序列 + 对齐路径。
    返回: (aligned_seq1, aligned_seq2, path)"""
    n, m = len(seq1), len(seq2)
    if window is None:
        window = max(n, m) // 4
    window = max(window, abs(n - m))

    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    traceback = np.zeros((n + 1, m + 1, 2), dtype=int)

    for i in range(1, n + 1):
        for j in range(max(1, i - window), min(m, i + window) + 1):
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            candidates = [(dtw[i - 1, j], i - 1, j),
                          (dtw[i, j - 1], i, j - 1),
                          (dtw[i - 1, j - 1], i - 1, j - 1)]
            min_val, pi, pj = min(candidates, key=lambda x: x[0])
            dtw[i, j] = cost + min_val
            traceback[i, j] = [pi, pj]

    # 回溯对齐路径
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 0 and j > 0:
        i, j = traceback[i, j]
        if i > 0 or j > 0:
            path.append((max(0, i - 1), max(0, j - 1)))
    path.reverse()

    aligned1 = np.array([seq1[p[0]] for p in path])
    aligned2 = np.array([seq2[p[1]] for p in path])
    return aligned1, aligned2, path


# ----------------------------------------------------------------
# DTW-KNN 分类器
# ----------------------------------------------------------------

class DTWKNN:
    """基于 DTW 距离的 KNN 分类器。

    Parameters:
        n_neighbors: K 值
        window: DTW 窗口约束
        downsample: 是否先下采样到固定帧数再 DTW（加快速度）
        target_frames: 下采样目标帧数
    """

    def __init__(self, n_neighbors=5, window=None, downsample=True, target_frames=30):
        self.n_neighbors = n_neighbors
        self.window = window
        self.downsample = downsample
        self.target_frames = target_frames
        self.templates = []   # list of (sequence, label_id)
        self.labels_ = None

    def fit(self, sequences, labels):
        """存储所有模板序列。

        参数:
            sequences: list of np.array, 每个形状 (T_i, 126)
            labels: list of int, 对应的标签 ID
        """
        for seq, label in zip(sequences, labels):
            processed = self._preprocess(seq)
            self.templates.append((processed, label))
        self.labels_ = np.array([l for _, l in self.templates])

    def _preprocess(self, sequence):
        """预处理序列：平滑 → 归一化 → 可选下采样。"""
        if len(sequence) < 2:
            return sequence
        seq = smooth_sequence(sequence, alpha=0.4)
        seq = normalize_sequence(seq)
        if self.downsample:
            seq = downsample_sequence(seq, self.target_frames)
        return seq

    def predict(self, sequence):
        """预测单个序列的标签。"""
        processed = self._preprocess(sequence)
        distances = []
        for tmpl_seq, label in self.templates:
            dist = dtw_distance(processed, tmpl_seq, self.window)
            distances.append((dist, label))
        distances.sort(key=lambda x: x[0])
        k = min(self.n_neighbors, len(distances))
        nearest = [label for _, label in distances[:k]]
        return Counter(nearest).most_common(1)[0][0]

    def predict_proba(self, sequence):
        """预测并返回各类别的概率分布。"""
        processed = self._preprocess(sequence)
        distances = []
        for tmpl_seq, label in self.templates:
            dist = dtw_distance(processed, tmpl_seq, self.window)
            distances.append((dist, label))

        distances.sort(key=lambda x: x[0])
        k = min(self.n_neighbors, len(distances))
        nearest = distances[:k]

        # DTW 距离越大相似度越低，用 1/(d+ε) 做权重
        n_classes = int(max(self.labels_)) + 1
        scores = np.zeros(n_classes)
        for dist, label in nearest:
            weight = 1.0 / (dist + 1e-6)
            scores[label] += weight
        proba = scores / (scores.sum() + 1e-8)
        return proba

    def predict_with_confidence(self, sequence, threshold=0.5):
        """预测并返回 (label_id, confidence, 是否超过阈值)。"""
        proba = self.predict_proba(sequence)
        pred = int(np.argmax(proba))
        conf = float(proba[pred])
        return pred, conf, conf >= threshold


# ----------------------------------------------------------------
# 运动检测（实时分割用）
# ----------------------------------------------------------------

def compute_motion_energy(keypoints_frame):
    """计算单帧关键点的'运动能量'（手腕偏离原点的程度）。
    用于快速判断帧内是否有手。"""
    lh_present = not np.allclose(keypoints_frame[0:3], 0)
    rh_present = not np.allclose(keypoints_frame[63:66], 0)
    if not lh_present and not rh_present:
        return None  # 没有手
    return lh_present, rh_present


class MotionSegmenter:
    """基于手腕位移的运动分割器。

    工作原理: 维护一个滑动窗口缓冲区，通过检测手腕的帧间位移
    来判断手势的起止。当位移超过 onset_threshold 时开始积累帧，
    当位移持续低于 offset_threshold 时结束并返回分割出的序列。
    """

    def __init__(self, buffer_seconds=3.0, fps=30,
                 onset_threshold=0.015, offset_threshold=0.008,
                 min_gesture_frames=10, max_gesture_frames=90,
                 settle_frames=15):
        self.buffer_size = int(buffer_seconds * fps)
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.min_gesture_frames = min_gesture_frames
        self.max_gesture_frames = max_gesture_frames
        self.settle_frames = settle_frames

        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.keypoint_buffer = deque(maxlen=self.buffer_size)

        self.is_gesturing = False
        self.gesture_frames = []
        self.still_count = 0
        self.last_wrist_pos = None

    def update(self, keypoints):
        """处理新一帧的关键点。

        参数:
            keypoints: (126,) 数组

        返回:
            None (没有检测到完整手势)
            或 np.array (形状 (T, 126)) 当检测到一个完整手势时
        """
        self.keypoint_buffer.append(keypoints)
        self.frame_buffer.append(keypoints)

        # 检测有手存在
        has_hand = not np.allclose(keypoints[0:3], 0) or not np.allclose(keypoints[63:66], 0)
        if not has_hand:
            self.is_gesturing = False
            self.gesture_frames = []
            self.still_count = 0
            self.last_wrist_pos = None
            return None

        # 计算手腕位置（使用活跃手）
        if not np.allclose(keypoints[0:3], 0):
            wrist = keypoints[0:3].copy()
        else:
            wrist = keypoints[63:66].copy()

        if self.last_wrist_pos is None:
            self.last_wrist_pos = wrist
            return None

        displacement = np.linalg.norm(wrist - self.last_wrist_pos)
        self.last_wrist_pos = wrist

        if not self.is_gesturing:
            if displacement > self.onset_threshold:
                self.is_gesturing = True
                self.gesture_frames = [keypoints.copy()]
                self.still_count = 0
        else:
            self.gesture_frames.append(keypoints.copy())

            if len(self.gesture_frames) >= self.max_gesture_frames:
                # 达到最大长度，强制截断
                seq = np.array(self.gesture_frames)
                self._reset()
                if len(seq) >= self.min_gesture_frames:
                    return seq
                return None

            if displacement < self.offset_threshold:
                self.still_count += 1
                if self.still_count >= self.settle_frames:
                    # 静止足够久，手势结束
                    seq = np.array(self.gesture_frames[:-self.settle_frames])
                    self._reset()
                    if len(seq) >= self.min_gesture_frames:
                        return seq
                    return None
            else:
                self.still_count = 0

        return None

    def _reset(self):
        self.is_gesturing = False
        self.gesture_frames = []
        self.still_count = 0
        self.last_wrist_pos = None
