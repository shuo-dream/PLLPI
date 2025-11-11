import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import re

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def parse_metrics(file_path, model_name):
    """
    解析指标文件，提取测试集平均指标
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取测试集平均指标
    test_metrics_section = re.search(r'测试集平均指标:\n(.*?)\n\n', content, re.DOTALL)
    if not test_metrics_section:
        raise ValueError(f"无法在 {file_path} 中找到测试集平均指标")
    
    metrics_text = test_metrics_section.group(1)
    metrics = {}
    for line in metrics_text.strip().split('\n'):
        key, value = line.split(':')
        metrics[key.strip()] = float(value.strip())
    
    return metrics

# 定义四个模型的文件路径和名称
models = {
    'PLLPI': 'G:\\shen_cong\\my\\my_project\\compare\\PLLPI',
    'PLLPI_PL_A': 'G:\\shen_cong\\my\\my_project\\compare\\PLLPI_PL_A',
    'PLLPI_PL_B': 'G:\\shen_cong\\my\\my_project\\compare\\PLLPI_PL_B',
    'PLLPI_PL_C': 'G:\\shen_cong\\my\\my_project\\compare\\PLLPI_PL_C'
}

# 提取所有模型的测试集指标
all_metrics = {}
for model_name, file_path in models.items():
    all_metrics[model_name] = parse_metrics(file_path, model_name)

# 准备绘图数据
metrics_names = list(all_metrics['PLLPI'].keys())  # Accuracy, Precision, Recall, F1, Auc, Aupr
model_names = list(models.keys())

# 创建图形和子图
fig, ax = plt.subplots(figsize=(12, 8))

# 设置柱状图位置
x = np.arange(len(metrics_names))  # 标签位置
width = 0.2  # 柱状图宽度

# 绘制每个模型的柱状图
bars = []
for i, model_name in enumerate(model_names):
    offset = (i - len(model_names)/2 + 0.5) * width
    values = [all_metrics[model_name][metric] for metric in metrics_names]
    bar = ax.bar(x + offset, values, width, label=model_name)
    bars.append(bar)

# 添加标签和标题
ax.set_xlabel('评价指标')
ax.set_ylabel('数值')
ax.set_title('四个模型测试集平均指标对比')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)

# 调整图例位置以避免遮挡
legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

# 在柱状图上显示数值
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for bar_group in bars:
    add_value_labels(bar_group)

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 保存图形到文件
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为 model_comparison.png")