##添加时间代码；去掉多余printing结果；增加可视化
##需要统一！同时注意时间代码：只计时方法特有的核心变换部分，而所有方法共有的步骤（数据加载、特征选择、模型训练等）应不计入方法耗时，或统一采用相同的实现。
##Spline 方法核心耗时步骤是“find_optimal_n_basis_with_viz” & “extract_spline_features_fixed”， 但是还有参数选择（但是其他方法可能不包括），需要最终确定是否包括参数选择时间。
##该代码的时间报告包括：
# 1.单独参数计算步骤的时间（n_basis selection time）；
# 2.样条方法不加上参数计算的时间（spline_feature_time）；
# 3.样条方法加上参数计算的时间（total_spline_time）；
# 4.总体运行时间（从main开始到结束的总时间，没有）

import numpy as np
import pandas as pd
import os
import time
import warnings
import seaborn as sns
from scipy.interpolate import BSpline
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, mean_squared_error, r2_score
)
from sklearn.preprocessing import StandardScaler, label_binarize
import tracemalloc
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt



warnings.filterwarnings('ignore')

# ============== 路径配置 ==============
ORIGINAL_ROOT = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/UCI HAR Dataset"
NEW_ROOT = "/Users/augleovo/PycharmProjects/PythonProject spline/.venv/bin/Integrated_HAR_Dataset_Sampled"

# ============== 数据预处理（静默执行）==============
def save_to_uci_format(y, sub, cts, fun_dict, split_name):
    path = os.path.join(NEW_ROOT, split_name)
    inert_path = os.path.join(path, "Inertial Signals")
    os.makedirs(inert_path, exist_ok=True)
    np.savetxt(os.path.join(path, f"y_{split_name}.txt"), y, fmt='%d')
    np.savetxt(os.path.join(path, f"subject_{split_name}.txt"), sub, fmt='%d')
    np.savetxt(os.path.join(path, f"X_{split_name}.txt"), cts, fmt='%.8f')
    names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
             'body_acc_x', 'body_acc_y', 'body_acc_z',
             'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    for j, name in enumerate(names):
        np.savetxt(os.path.join(inert_path, f"{name}_{split_name}.txt"),
                   fun_dict[j], fmt='%.8f')

def process_split(split):
    p = os.path.join(ORIGINAL_ROOT, split)
    y = pd.read_csv(os.path.join(p, f"y_{split}.txt"), header=None, delim_whitespace=True)[0].values
    sub = pd.read_csv(os.path.join(p, f"subject_{split}.txt"), header=None, delim_whitespace=True)[0].values
    cts = pd.read_csv(os.path.join(p, f"X_{split}.txt"), delim_whitespace=True, header=None).values
    sig_names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                 'body_acc_x', 'body_acc_y', 'body_acc_z',
                 'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    fun_signals = []
    for sig in sig_names:
        file_path = os.path.join(p, "Inertial Signals", f"{sig}_{split}.txt")
        fun_signals.append(pd.read_csv(file_path, delim_whitespace=True, header=None).values)
    idx_sampled = np.arange(0, len(y), 3)
    y_sampled = y[idx_sampled]
    sub_sampled = sub[idx_sampled]
    cts_sampled = cts[idx_sampled]
    fun_sampled = [sig[idx_sampled] for sig in fun_signals]
    save_to_uci_format(y_sampled, sub_sampled, cts_sampled,
                       {j: fun_sampled[j] for j in range(9)}, split)

def preprocess_har_data():
    if not os.path.exists(NEW_ROOT):
        print("Preprocessing: generating step-3 sampled dataset...")
        process_split("train")
        process_split("test")
    else:
        pass  # 静默跳过

# ============== 数据加载 ==============
def load_har_data_base(split="train", n_samples=None, p_cts=50, random_state=42):
    base_path = NEW_ROOT
    y = pd.read_csv(f"{base_path}/{split}/y_{split}.txt", header=None, delim_whitespace=True)[0].values
    indices = np.arange(len(y))
    if n_samples is not None:
        indices = indices[:min(n_samples, len(indices))]
    X_cts = pd.read_csv(f"{base_path}/{split}/X_{split}.txt", header=None, delim_whitespace=True).iloc[indices, :p_cts].values
    return indices, y[indices], X_cts, base_path

def load_har_for_spline(split="train", n_samples=None, p_cts=50, random_state=42):
    res = load_har_data_base(split, n_samples, p_cts, random_state)
    final_indices, y, X_cts, base_path = res
    signal_names = ['total_acc_x', 'total_acc_y', 'total_acc_z',
                    'body_acc_x', 'body_acc_y', 'body_acc_z',
                    'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
    X_func = np.zeros((len(final_indices), len(signal_names), 128))
    for j, sig in enumerate(signal_names):
        data = pd.read_csv(f"{base_path}/{split}/Inertial Signals/{sig}_{split}.txt",
                           header=None, delim_whitespace=True).values
        X_func[:, j, :] = data[final_indices]
    return {'X_func': X_func, 'X_cts': X_cts, 'y': y, 'time_grid': np.linspace(0, 1, 128)}

# ============== n_basis 选择（保留可视化，去掉print）==============
def find_optimal_n_basis_with_viz(X_func, time_grid, degree=3, max_basis=25):
    np.random.seed(42)
    n_samples, n_signals, n_points = X_func.shape
    sample_idx = np.random.choice(n_samples, min(20, n_samples), replace=False)
    candidate_range = range(degree + 2, min(max_basis, n_points // 3))
    avg_mses = []
    std_mses = []
    for n in candidate_range:
        n_internal_knots = n - degree - 1
        if n_internal_knots < 1:
            avg_mses.append(np.inf)
            std_mses.append(0)
            continue
        knots = np.concatenate((
            [time_grid[0]] * (degree + 1),
            np.linspace(time_grid[1], time_grid[-2], n_internal_knots),
            [time_grid[-1]] * (degree + 1)
        ))
        mses = []
        for idx in sample_idx:
            for sig_ch in range(min(3, n_signals)):
                signal = X_func[idx, sig_ch, :]
                try:
                    design_matrix = np.zeros((n_points, n))
                    for i in range(n):
                        coeffs = np.zeros(n); coeffs[i] = 1.0
                        spl_basis = BSpline(knots, coeffs, degree)
                        design_matrix[:, i] = spl_basis(time_grid)
                    pseudo_inv = np.linalg.pinv(design_matrix)
                    coeffs = pseudo_inv @ signal
                    reconstructed = design_matrix @ coeffs
                    mse = np.mean((signal - reconstructed) ** 2)
                    mses.append(mse)
                except:
                    mses.append(np.inf)
        avg_mses.append(np.mean(mses))
        std_mses.append(np.std(mses))
    diffs = np.diff(avg_mses)
    second_diffs = np.diff(diffs)
    if len(second_diffs) > 0:
        elbow_idx = np.argmin(second_diffs) + 1
    else:
        elbow_idx = len(candidate_range) // 2
    optimal_n = list(candidate_range)[min(elbow_idx, len(candidate_range) - 1)]
    # 绘图仍显示，但不输出文字
    plt.figure(figsize=(10, 5))
    plt.errorbar(candidate_range, avg_mses, yerr=std_mses, fmt='o-', capsize=5,
                 label='Reconstruction Error')
    plt.axvline(x=optimal_n, color='r', linestyle='--',
                label=f'Optimal n_basis={optimal_n}')
    plt.xlabel('Number of Basis Functions (n_basis)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Elbow Method for n_basis Selection')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.show()
    return optimal_n

# ============== 提取样条特征（去掉print）==============
def extract_spline_features_fixed(X_func, time_grid, n_basis, degree=3):
    n_samples, n_signals, n_points = X_func.shape
    n_internal_knots = n_basis - degree - 1
    if n_internal_knots < 1:
        n_internal_knots = 1
        n_basis = n_internal_knots + degree + 1
    knots = np.concatenate((
        [time_grid[0]] * (degree + 1),
        np.linspace(time_grid[0], time_grid[-1], n_internal_knots + 2)[1:-1],
        [time_grid[-1]] * (degree + 1)
    ))
    design_matrix = np.zeros((n_points, n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis); coeffs[i] = 1.0
        spl_basis = BSpline(knots, coeffs, degree)
        design_matrix[:, i] = spl_basis(time_grid)
    pseudo_inv = np.linalg.pinv(design_matrix)
    spline_features = np.zeros((n_samples, n_signals * n_basis))
    for i in range(n_samples):
        sample_all_sigs = []
        for j in range(n_signals):
            coeffs = pseudo_inv @ X_func[i, j, :]
            sample_all_sigs.extend(coeffs)
        spline_features[i, :] = sample_all_sigs
    return spline_features



# ============== 样条拟合可视化 ==============
def plot_spline_fitting_example(X_func, time_grid, n_basis, degree=3, sample_idx=0, signal_ch=0):
    # 使用同样的方法计算重构信号
    n_internal_knots = n_basis - degree - 1
    knots = np.concatenate((
        [time_grid[0]] * (degree + 1),
        np.linspace(time_grid[0], time_grid[-1], n_internal_knots + 2)[1:-1],
        [time_grid[-1]] * (degree + 1)
    ))
    design_matrix = np.zeros((len(time_grid), n_basis))
    for i in range(n_basis):
        coeffs = np.zeros(n_basis);
        coeffs[i] = 1.0
        spl_basis = BSpline(knots, coeffs, degree)
        design_matrix[:, i] = spl_basis(time_grid)
    pseudo_inv = np.linalg.pinv(design_matrix)

    signal = X_func[sample_idx, signal_ch, :]
    coeffs = pseudo_inv @ signal
    reconstructed = design_matrix @ coeffs

    plt.figure(figsize=(10, 4))
    plt.plot(time_grid, signal, 'o-', markersize=2, label='Original (128 points)')
    plt.plot(time_grid, reconstructed, 'r-', linewidth=2, label=f'Spline Reconstruction (n_basis={n_basis})')
    plt.xlabel('Normalized Time')
    plt.ylabel('Acceleration')
    plt.title(f'Sample {sample_idx}, Signal {signal_ch} - B-spline fitting')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

#对比不同基函数数量的样条拟合可视化函数
def plot_spline_comparison(X_func, time_grid, n_basis_list, degree=3, sample_idx=0, signal_ch=0):
    """
    对比不同基函数数量下的样条拟合效果。
    参数：
        n_basis_list : list[int]  要对比的基函数数量列表，例如 [8, 14, 20]
    """
    signal = X_func[sample_idx, signal_ch, :]

    plt.figure(figsize=(12, 5))
    # 绘制原始信号（灰色半透明，突出对比）
    plt.plot(time_grid, signal, 'o-', markersize=2, color='black', alpha=0.4, label='Original (128 points)')

    # 为不同 n_basis 分配不同颜色
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(n_basis_list)))

    for n, color in zip(n_basis_list, colors):
        # 使用与 extract_spline_features_fixed 完全相同的重构逻辑
        n_internal_knots = n - degree - 1
        if n_internal_knots < 1:
            n_internal_knots = 1
            n = n_internal_knots + degree + 1

        knots = np.concatenate((
            [time_grid[0]] * (degree + 1),
            np.linspace(time_grid[0], time_grid[-1], n_internal_knots + 2)[1:-1],
            [time_grid[-1]] * (degree + 1)
        ))

        design_matrix = np.zeros((len(time_grid), n))
        for i in range(n):
            coeffs = np.zeros(n)
            coeffs[i] = 1.0
            spl_basis = BSpline(knots, coeffs, degree)
            design_matrix[:, i] = spl_basis(time_grid)

        pseudo_inv = np.linalg.pinv(design_matrix)
        coeffs = pseudo_inv @ signal
        reconstructed = design_matrix @ coeffs
        mse = np.mean((signal - reconstructed) ** 2)

        plt.plot(time_grid, reconstructed, color=color, linewidth=2,
                 label=f'n_basis={n} (MSE={mse:.4f})')

    plt.xlabel('Normalized Time')
    plt.ylabel('Acceleration')
    plt.title(f'Sample {sample_idx}, Signal {signal_ch} - B-spline Fitting Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============== 多模型比较（仅保留你想要的输出）==============
def compare_models_with_pipeline(X_train, y_train, X_test, y_test, opt_k):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=None,
                                                min_samples_split=2, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    }
    results = []
    fitted_pipes = {}
    print("\nComparing different classifiers:")
    for name, clf in models.items():
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(score_func=mutual_info_classif, k=opt_k)),
            ('clf', clf)
        ])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=-1)
        pipe.fit(X_train, y_train)
        test_acc = pipe.score(X_test, y_test)
        y_pred = pipe.predict(X_test)
        y_pred_proba = pipe.predict_proba(X_test)

        # Classification metrics
        prec_macro = precision_score(y_test, y_pred, average='macro')
        rec_macro = recall_score(y_test, y_pred, average='macro')
        f1_mac = f1_score(y_test, y_pred, average='macro')
        prec_weighted = precision_score(y_test, y_pred, average='weighted')
        rec_weighted = recall_score(y_test, y_pred, average='weighted')
        f1_wt = f1_score(y_test, y_pred, average='weighted')

        # ROC-AUC
        classes_list = sorted(np.unique(y_train))
        y_test_bin = label_binarize(y_test, classes=classes_list)
        macro_auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovr', average='macro')

        # Regression-style metrics
        mse_val = mean_squared_error(y_test, y_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_test, y_pred)

        results.append({
            'Model': name,
            'CV Mean': np.mean(cv_scores),
            'CV Std': np.std(cv_scores),
            'Test Acc': test_acc,
            'Precision (Macro)': prec_macro,
            'Recall (Macro)': rec_macro,
            'F1 (Macro)': f1_mac,
            'Precision (Weighted)': prec_weighted,
            'Recall (Weighted)': rec_weighted,
            'F1 (Weighted)': f1_wt,
            'ROC-AUC (Macro)': macro_auc,
            'MSE': mse_val,
            'RMSE': rmse_val,
            'R²': r2_val
        })
        fitted_pipes[name] = pipe

        print(f"\n{name}:")
        print(f"  CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        print(f"  Test Accuracy: {test_acc:.4f}")
        print(f"  Precision/Recall/F1 (Macro): {prec_macro:.4f} / {rec_macro:.4f} / {f1_mac:.4f}")
        print(f"  ROC-AUC (Macro): {macro_auc:.4f}")
        print(f"  MSE: {mse_val:.4f}  RMSE: {rmse_val:.4f}  R²: {r2_val:.4f}")
    return pd.DataFrame(results), fitted_pipes

# ============== 主流程（精简输出）==============
def main():
    np.random.seed(42)
    overall_start = time.time()
    tracemalloc.start()


    # 静默预处理
    preprocess_har_data()

    # 加载数据
    train_data = load_har_for_spline("train", n_samples=None, random_state=42)
    test_data = load_har_for_spline("test", n_samples=None, random_state=42)


    # ---------- 样条特有部分：参数选择 ----------
    n_basis_start = time.time()
    # 选择 n_basis
    opt_n_basis = find_optimal_n_basis_with_viz(
        train_data['X_func'], train_data['time_grid'], degree=3, max_basis=25
    )
    n_basis_time = time.time() - n_basis_start

    # ---------- 样条特有部分：特征提取 ----------
    spline_start = time.time()
    # 提取样条特征
    X_train_spl = extract_spline_features_fixed(
        train_data['X_func'], train_data['time_grid'], opt_n_basis
    )
    X_test_spl = extract_spline_features_fixed(
        test_data['X_func'], test_data['time_grid'], opt_n_basis
    )
    spline_feature_time = time.time() - spline_start

    # ============== 样条拟合可视化 ==============
    print("\nShowing spline fitting example...")  # 可选提示
    plot_spline_fitting_example(
        X_func=train_data['X_func'],
        time_grid=train_data['time_grid'],
        n_basis=opt_n_basis,
        degree=3,
        sample_idx=0,
        signal_ch=0
    )

    # ============== 对比不同基函数数量的拟合效果 ==============
    # 选取代表性的样本和信号（例如第0个样本、第0个信号）
    # 基函数数量选择 [8, opt_n_basis, 20]
    candidate_n_list = [8, opt_n_basis, 20]
    # 确保所有值不超过 max_basis 且大于 degree+1
    candidate_n_list = [n for n in candidate_n_list if n <= 25 and n >= 5]
    print("\nShowing spline fitting comparison...")
    plot_spline_comparison(
        X_func=train_data['X_func'],
        time_grid=train_data['time_grid'],
        n_basis_list=candidate_n_list,
        degree=3,
        sample_idx=0,
        signal_ch=0
    )


    # 合并特征
    X_train_combined = np.hstack([X_train_spl, train_data['X_cts']])
    X_test_combined = np.hstack([X_test_spl, test_data['X_cts']])

    # 动态特征选择
    total_feats = X_train_combined.shape[1]
    base_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=mutual_info_classif)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_grid = {
        'select__k': np.unique([
            20, total_feats // 4, total_feats // 3,
            total_feats // 2, total_feats * 2 // 3
        ]).astype(int)
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(base_pipe, param_grid=param_grid,
                               cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_combined, train_data['y'])
    opt_k = grid_search.best_params_['select__k']

    # 模型比较与输出（核心输出）
    results_df, fitted_pipes = compare_models_with_pipeline(
        X_train_combined, train_data['y'],
        X_test_combined, test_data['y'], opt_k
    )

    # 摘要
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    best_model = results_df.loc[results_df['Test Acc'].idxmax()]
    print(f"\nBest Model: {best_model['Model']}")
    print(f"Test Accuracy: {best_model['Test Acc']:.4f}")
    print(f"CV Accuracy: {best_model['CV Mean']:.4f} (+/- {best_model['CV Std']:.4f})")

    # 用best model的fitted pipeline做预测
    best_pipe = fitted_pipes[best_model['Model']]
    y_pred = best_pipe.predict(X_test_combined)
    y_pred_proba = best_pipe.predict_proba(X_test_combined)

    target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs',
                    'Sitting', 'Standing', 'Laying']

    # 时间报告
    print("\n" + "=" * 60)
    print("TIME BENCHMARK (Spline Method)")
    print("=" * 60)
    print(f"n_basis selection time:      {n_basis_time:.4f} sec")
    print(f"Spline feature extraction:   {spline_feature_time:.4f} sec")
    print(f"Total spline-specific time:  {n_basis_time + spline_feature_time:.4f} sec")
    print("=" * 60)

    # ---- Memory usage ----
    mem_current, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nMemory Usage:")
    print(f"  Current:          {mem_current / 1024 / 1024:.2f} MB")
    print(f"  Peak:             {mem_peak / 1024 / 1024:.2f} MB")

    # ======== Visualisation ========
    # 1. Confusion Matrix Heatmap
    cm = confusion_matrix(test_data['y'], y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Walking', 'WalkUp', 'WalkDown', 'Sitting', 'Standing', 'Laying'],
                yticklabels=['Walking', 'WalkUp', 'WalkDown', 'Sitting', 'Standing', 'Laying'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {best_model["Model"]}')
    plt.tight_layout()
    plt.savefig('spline_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 2. ROC Curves (One-vs-Rest)
    classes_list = sorted(np.unique(train_data['y']))
    y_test_bin = label_binarize(test_data['y'], classes=classes_list)
    activity_names = {1: 'WALKING', 2: 'WALKING_UP', 3: 'WALKING_DOWN',
                      4: 'SITTING', 5: 'STANDING', 6: 'LAYING'}

    fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
    colors_roc = plt.cm.tab10(np.linspace(0, 1, len(classes_list)))
    auc_scores = []

    for i, (cl, color) in enumerate(zip(classes_list, colors_roc)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        auc_val = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
        auc_scores.append(auc_val)
        ax_roc.plot(fpr, tpr, color=color, linewidth=2,
                    label=f'{activity_names[cl]} (AUC = {auc_val:.3f})')

    macro_auc = np.mean(auc_scores)
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax_roc.set_xlabel('False Positive Rate', fontsize=12)
    ax_roc.set_ylabel('True Positive Rate', fontsize=12)
    ax_roc.set_title(f'ROC Curves (One-vs-Rest) - {best_model["Model"]}\n'
                     f'Macro AUC: {macro_auc:.3f}',
                     fontsize=14, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10)
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('spline_roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Classification report
    print(f"\nDetailed classification report for {best_model['Model']}:")
    print(classification_report(
        test_data['y'], y_pred,
        target_names=target_names,
        digits=4
    ))

    # ======== Save all results to CSV ========
    print("\n--- Saving results ---")

    # 1. Model comparison CSV (all 3 classifiers)
    results_df['Peak Memory (MB)'] = mem_peak / 1024 / 1024
    results_df['n_basis'] = opt_n_basis
    results_df['k_features'] = opt_k
    results_df['n_basis_time (s)'] = n_basis_time
    results_df['spline_feature_time (s)'] = spline_feature_time
    results_df['total_spline_time (s)'] = n_basis_time + spline_feature_time
    results_df.to_csv('spline_model_comparison.csv', index=False)
    print("Saved: spline_model_comparison.csv")

    # 2. Best model detailed metrics CSV
    best_metrics = pd.DataFrame({
        'Metric': ['Best Model', 'Accuracy', 'Precision (Macro)', 'Recall (Macro)',
                   'F1-Score (Macro)', 'Precision (Weighted)', 'Recall (Weighted)',
                   'F1-Score (Weighted)', 'ROC-AUC (Macro)',
                   'MSE', 'RMSE', 'R2 Score',
                   'n_basis', 'k_features', 'Total Features',
                   'n_basis Selection Time (s)', 'Spline Feature Time (s)',
                   'Total Spline Time (s)', 'Peak Memory (MB)'],
        'Value': [best_model['Model'], best_model['Test Acc'],
                  best_model['Precision (Macro)'], best_model['Recall (Macro)'],
                  best_model['F1 (Macro)'], best_model['Precision (Weighted)'],
                  best_model['Recall (Weighted)'], best_model['F1 (Weighted)'],
                  best_model['ROC-AUC (Macro)'],
                  best_model['MSE'], best_model['RMSE'], best_model['R²'],
                  opt_n_basis, opt_k, total_feats,
                  n_basis_time, spline_feature_time,
                  n_basis_time + spline_feature_time,
                  mem_peak / 1024 / 1024]
    })
    best_metrics.to_csv('spline_best_model_results.csv', index=False)
    print("Saved: spline_best_model_results.csv")

    # 3. Per-class AUC CSV
    auc_df = pd.DataFrame({
        'Class': [activity_names[cl] for cl in classes_list],
        'AUC': auc_scores
    })
    auc_df.loc[len(auc_df)] = ['Macro Average', macro_auc]
    auc_df.to_csv('spline_per_class_auc.csv', index=False)
    print("Saved: spline_per_class_auc.csv")

    return {
        'n_basis': opt_n_basis,
        'k_features': opt_k,
        'best_model': best_model['Model'],
        'test_accuracy': best_model['Test Acc'],
        'cv_accuracy': best_model['CV Mean'],
        'total_features': total_feats,
        'results': results_df,
        'n_basis_time': n_basis_time,
        'spline_feature_time': spline_feature_time,
        'total_spline_time': n_basis_time + spline_feature_time,
        'peak_memory_mb': mem_peak / 1024 / 1024
    }




if __name__ == "__main__":
    results = main()
