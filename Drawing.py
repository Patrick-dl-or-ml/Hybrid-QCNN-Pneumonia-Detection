import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==========================================
# 1. 读取你挂机跑出来的 CSV 数据
# ==========================================
csv_file = "experiment_30_rounds_results.csv"

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"❌ 找不到文件 {csv_file}！请确保 30 轮训练已经跑完并生成了该文件。")
    exit()

# 分离两种模型的数据
classical_data = df[df['Model_Type'] == 'Classical_CNN']
hybrid_data = df[df['Model_Type'] == 'Hybrid_QCNN']


# ==========================================
# 2. 计算均值和 p-value (严格对标论文 Table 1)
# ==========================================
# 论文作者使用了单尾 Welsch's t-test (方差不相等的 t 检验)
def calculate_metrics(metric_name, is_loss=False):
    c_mean = classical_data[metric_name].mean()
    h_mean = hybrid_data[metric_name].mean()

    # 对于 Loss，我们期望 Hybrid 更小；对于 Acc/AUC，期望 Hybrid 更大
    alt = 'greater' if is_loss else 'less'

    # 计算 p-value
    _, p_val = stats.ttest_ind(
        classical_data[metric_name],
        hybrid_data[metric_name],
        equal_var=False,
        alternative=alt
    )
    return c_mean, h_mean, p_val


# 提取四项核心指标
c_train_loss, h_train_loss, p_train_loss = calculate_metrics('Train_Loss', is_loss=True)
c_val_loss, h_val_loss, p_val_loss = calculate_metrics('Val_Loss', is_loss=True)
c_test_acc, h_test_acc, p_test_acc = calculate_metrics('Test_Accuracy', is_loss=False)
c_test_auc, h_test_auc, p_test_auc = calculate_metrics('Test_AUROC', is_loss=False)

# ==========================================
# 3. 在终端打印完美的 Table 1
# ==========================================
print("\n" + "=" * 70)
print(f"{'Metric':<30} | {'Classical':<12} | {'Hybrid':<12} | {'p-value':<10}")
print("-" * 70)
print(f"{'Mean loss on training set':<30} | {c_train_loss:<12.3f} | {h_train_loss:<12.3f} | {p_train_loss:<10.3f}")
print(f"{'Mean loss on validation set':<30} | {c_val_loss:<12.3f} | {h_val_loss:<12.3f} | {p_val_loss:<10.3f}")
print(f"{'AUROC on test set':<30} | {c_test_auc:<12.3f} | {h_test_auc:<12.3f} | {p_test_auc:<10.3f}")
print(f"{'Accuracy on test set':<30} | {c_test_acc:<12.3f} | {h_test_acc:<12.3f} | {p_test_acc:<10.3f}")
print("=" * 70)

# ==========================================
# 4. 画图：直观的性能对比图 (Accuracy & AUROC)
# ==========================================
labels = ['Test Accuracy', 'Test AUROC']
classical_means = [c_test_acc, c_test_auc]
hybrid_means = [h_test_acc, h_test_auc]

x = [0, 1]
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar([i - width / 2 for i in x], classical_means, width, label='Classical CNN', color='#1f77b4')
rects2 = ax.bar([i + width / 2 for i in x], hybrid_means, width, label='Hybrid QCNN', color='#ff7f0e')

ax.set_ylabel('Scores')
ax.set_title('Performance Comparison (Averaged over 30 rounds)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower right')
ax.set_ylim([0.5, 1.0])  # 设置Y轴范围更好看

# 在柱子上标出具体数值
ax.bar_label(rects1, fmt='%.3f', padding=3)
ax.bar_label(rects2, fmt='%.3f', padding=3)

fig.tight_layout()
plt.savefig("performance_comparison.png", dpi=300)
print("\n✅ 对比柱状图已保存为 performance_comparison.png！")
plt.show()