"""
手语分类模型训练
- 加载采集的关键点数据
- 训练分类器（sklearn MLP + 可选 KNN）
- 评估准确率并保存模型
"""
import numpy as np
import os
import json
import pickle
import sys

sys.path.insert(0, os.path.dirname(__file__))


def load_data(data_dir="training_data"):
    """加载训练数据"""
    data_file = os.path.join(data_dir, "keypoints.npy")
    labels_file = os.path.join(data_dir, "labels.npy")
    mapping_file = os.path.join(data_dir, "label_map.json")

    if not all(os.path.exists(f) for f in [data_file, labels_file, mapping_file]):
        print("数据文件不完整，请先运行 collect_data.py 采集数据")
        print(f"  需要: {data_file}, {labels_file}, {mapping_file}")
        return None, None, None

    X = np.load(data_file)
    y = np.load(labels_file)
    with open(mapping_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    print(f"加载数据: {X.shape[0]} 个样本, {X.shape[1]} 维特征, {len(label_map)} 个类别")
    print(f"标签映射: {label_map}")
    return X, y, label_map


def train_mlp(X_train, y_train, num_classes):
    """训练 MLP 神经网络分类器"""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size=32,
        learning_rate="adaptive",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=True,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def train_knn(X_train, y_train, num_classes):
    """训练 KNN 分类器（作为备选）"""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=min(5, len(X_train)), weights="distance")
    model.fit(X_scaled, y_train)
    return model, scaler


def evaluate(model, scaler, X, y, label_map):
    """评估模型并打印报告"""
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    print(f"\n训练集准确率: {acc:.2%}")

    # 交叉验证（至少需要每类2个样本）
    min_samples_per_class = min(np.bincount(y))
    if min_samples_per_class >= 2:
        cv = StratifiedKFold(n_splits=min(3, min_samples_per_class), shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
        print(f"交叉验证准确率: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    else:
        print("样本太少，跳过交叉验证")

    id_to_label = {v: k for k, v in label_map.items()}
    target_names = [id_to_label[i] for i in range(len(label_map))]

    print("\n--- 分类报告 ---")
    print(classification_report(y, y_pred, target_names=target_names, zero_division=0))

    return acc


def save_model(model, scaler, label_map, save_dir="models"):
    """保存模型和预处理"""
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "classifier.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(save_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print(f"\n模型已保存到: {save_dir}/")
    print(f"  - classifier.pkl")
    print(f"  - scaler.pkl")
    print(f"  - label_map.json")


def main():
    from sklearn.model_selection import train_test_split

    data_dir = "training_data"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    X, y, label_map = load_data(data_dir)
    if X is None:
        return

    num_classes = len(label_map)
    if num_classes < 2:
        print("至少需要采集 2 种不同的手势才能训练！")
        return

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

    print("\n>>> 训练 MLP 神经网络 <<<")
    model, scaler = train_mlp(X_train, y_train, num_classes)

    print("\n>>> 测试集评估 <<<")
    evaluate(model, scaler, X_test, y_test, label_map)

    save_model(model, scaler, label_map)


if __name__ == "__main__":
    main()
