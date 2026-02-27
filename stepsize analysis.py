## data has 50% overlap - Strong correlation violate the assumption of IID
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


# ============== 1. Data Loading ==============
def load_raw_inertial_data(base_path, split="train", signal="total_acc_x"):
    """
    加载最原始的信号数据，用于分析样本间的相似度
    """
    file_path = f"{base_path}/{split}/Inertial Signals/{signal}_{split}.txt"
    data = pd.read_csv(file_path, header=None, delim_whitespace=True).values
    return data


# ============== 2. Similarity Analysis by cosine similarity ==============
# cosine similarity: 一个样本是 128个连续的传感器数值。
# 人体动作的传感器信号具有极强的方向性。加速度的上下波动会形成特定的波形。如果两个窗口重叠了，它们的波形走势（方向）几乎是一致的。
# 能直接反映出相邻两个窗口在时域波形上的重叠程度，比较波形 -- cosine similarity
def analyze_step_similarity(data, max_step=10):
    """
    计算随着步长增大，相邻样本之间的余弦相似度变化 Calculate the change in cosine similarity between adjacent samples as the step size increases
    理由：如果相似度很高，说明样本不独立；相似度趋于稳定，说明达到了物理上的独立。
    """
    mean_similarities = []

    for s in range(1, max_step + 1):
        # 取相邻的样本对：(Row i) 和 (Row i+s)
        current_samples = data[:-s]
        next_samples = data[s:]

        # 计算每一对相邻样本的余弦相似度
        # 结果为 1 表示完全相同，0 表示正交（无相关）
        sims = [cosine_similarity(current_samples[i].reshape(1, -1),
                                  next_samples[i].reshape(1, -1))[0, 0]
                for i in range(len(current_samples))]

        mean_similarities.append(np.mean(sims))
        print(f"  Step {s}: Average Cosine Similarity = {mean_similarities[-1]:.4f}")

    return mean_similarities


# ============== 3. 泛化缺口测试 (验证 IID 假设) Generalization Gap ==============
# avoid data leakage(这会导致模型通过‘记忆’重叠的传感器波形来作弊，产生虚假的高准确率) caused be 50% data overlap.
# 在这个阶段是数据预处理，我们可以用test acc(?)，目的是为了诊断数据集的健康状态。真正不能的是在正式训练模型时。我不是在评估模型的最终性能，而是在进行‘数据泄露诊断’。
# 通过人为制造一个简单的 Train-Test 切分，观察在不同步长下，模型利用‘重叠信息’作弊的空间有多大。当 Gap（训练与测试的差距）从极小转为稳定时，说明我们成功切断了样本间的作弊路径。
def evaluate_generalization_gap(X_full, y_full, steps=[1, 2, 3, 5, 8]):
    """
    对比不同步长下，训练准确率和测试准确率的差距。
    按不同的步长（比如 Step 1, 2, 3）去抽样。让模型去学习，算出训练集准确率和测试集准确率。算这两者的差值（Gap）。
    理由：如果不独立（Step小），模型会通过记忆重叠数据点在测试集上作弊，Gap 会非常小。
    只有独立后，Gap 才会反映真实的泛化能力。
    """
    results = []

    for s in steps:
        # 按步长取样
        indices = np.arange(0, len(y_full), s)
        X_s = X_full[indices]
        y_s = y_full[indices]

        # 简单的 Train-Test Split (从原始的“X_train.txt”&“y_train.txt”文件中分割的-从后面main代码里显示)
        X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.3, random_state=42)

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, clf.predict(X_train))
        test_acc = accuracy_score(y_test, clf.predict(X_test))
        gap = train_acc - test_acc

        results.append({
            'step': s,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'gap': gap
        })
        print(f"  Step {s}: Train-Test Gap = {gap:.4f}")

    return pd.DataFrame(results)


# ============== 4. 绘图与结论输出 ==============
def plot_results(similarities, gap_df):
    plt.figure(figsize=(12, 5))

    # 子图 1: 物理相似度衰减
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(similarities) + 1), similarities, 'o-', color='teal')
    plt.axvline(x=2, color='red', linestyle='--', label='50% Overlap Limit')
    plt.title("Sample Similarity vs. Step Size")
    plt.xlabel("Step Size")
    plt.ylabel("Average Cosine Similarity")
    plt.legend()

    # 子图 2: 泛化缺口变化
    plt.subplot(1, 2, 2)
    plt.plot(gap_df['step'], gap_df['gap'], 's-', color='orange')
    plt.title("Generalization Gap (Train - Test)")
    plt.xlabel("Step Size")
    plt.ylabel("Accuracy Gap")

    plt.tight_layout()
    plt.show()


# ============== 执行分析 ==============
if __name__ == "__main__":
    # 请确保路径正确
    DATA_PATH = "UCI HAR Dataset"

    print("--- Phase 1: Analyzing Physical Signal Similarity ---")
    raw_acc = load_raw_inertial_data(DATA_PATH)
    sims = analyze_step_similarity(raw_acc)

    print("\n--- Phase 2: Analyzing Generalization Gap ---")
    # 为了快速分析，直接加载 X_train.txt (561维特征)--只用到了train文件在进行分割
    X_all = pd.read_csv(f"{DATA_PATH}/train/X_train.txt", header=None, delim_whitespace=True).values
    y_all = pd.read_csv(f"{DATA_PATH}/train/y_train.txt", header=None, delim_whitespace=True).values.ravel()

    gap_data = evaluate_generalization_gap(X_all, y_all)

    print("\n--- Phase 3: Visualizing Results ---")
    plot_results(sims, gap_data)




#结果：第一个图
#相似度差别看起来小原因：人体动作的连贯性
# 虽然相似度只下降了不到 1%，但这是在高维空间（128维）下的统计结果。这 1% 的下降代表了 50% 物理重叠的彻底消失。
# 剩下的 0.93 相似度不是因为‘重叠’，而是因为‘动作的一致性’。我们追求的是‘物理独立’，而不是‘波形互斥’。
#第二个图：
#step1-2 gap翻倍：因为在 Step 1 时，模型因为重叠数据靠作弊考了高分，所以 Gap 极小。
#2-3: 开始稳定 当 Gap 不再继续剧烈增长，而是开始在 4% 左右徘徊时，说明**“泄露”已经排干净了**。
# 剩下的 Gap 是由于模型正常的泛化能力不足引起的，而不是因为数据重叠。（stepsize太大，sample太少，数据得不到稳定的结果。））