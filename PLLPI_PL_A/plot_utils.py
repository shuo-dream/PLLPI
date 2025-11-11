import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm

# 全局中文字体属性
chinese_font_prop = None


# 设置中文字体支持
def init_chinese_font():
    """初始化中文字体"""
    global chinese_font_prop
    # 查找系统中的中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
    available_fonts = {f.name: f for f in fm.fontManager.ttflist}

    for font_name in chinese_fonts:
        if font_name in available_fonts:
            chinese_font_prop = fm.FontProperties(fname=available_fonts[font_name].fname)
            plt.rcParams['font.sans-serif'] = [font_name]
            break
    else:
        # 如果没有找到中文字体，使用默认字体并防止负号显示异常
        chinese_font_prop = fm.FontProperties()
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


# 在模块导入时初始化字体
init_chinese_font()


def get_chinese_font():
    """获取中文字体属性"""
    global chinese_font_prop
    return chinese_font_prop


def plot_individual_metrics(train_metrics_history, eval_metrics_history, save_dir=None):
    """
    为每个指标分别绘制训练集和测试集的对比曲线图

    Args:
        train_metrics_history: 训练集指标历史记录
        eval_metrics_history: 测试集指标历史记录
        save_dir: 图片保存目录（可选）
    """
    # 定义要绘制的指标
    metrics_info = {
        'accuracy': '准确率',
        'precision': '精确率',
        'recall': '召回率',
        'f1': 'F1分数',
        'auc': 'AUC',
        'aupr': 'AUPR'
    }

    # 为每个指标创建单独的图表
    for metric_key, metric_chinese in metrics_info.items():
        plt.figure(figsize=(10, 6))

        # 提取训练和测试指标值
        train_values = [m[metric_key] for m in train_metrics_history]
        eval_values = [m[metric_key] for m in eval_metrics_history]

        # 绘制训练和测试曲线
        epochs = range(1, len(train_values) + 1)
        plt.plot(epochs, train_values, label=f'训练{metric_chinese}', marker='o', linewidth=2)
        plt.plot(epochs, eval_values, label=f'测试{metric_chinese}', marker='s', linewidth=2)

        # 设置图表属性
        plt.title(f'{metric_chinese}变化曲线', fontsize=14, fontproperties=get_chinese_font())
        plt.xlabel('Epoch', fontproperties=get_chinese_font())
        plt.ylabel(metric_chinese, fontproperties=get_chinese_font())
        plt.legend(prop=get_chinese_font())
        plt.grid(True, alpha=0.3)

        # 设置y轴范围以更好地显示数据
        min_val = min(min(train_values), min(eval_values))
        max_val = max(max(train_values), max(eval_values))
        margin = (max_val - min_val) * 0.05  # 添加5%的边距
        if margin > 0:
            plt.ylim(max(0, min_val - margin), min(1, max_val + margin))

        # 保存图片（如果指定了保存路径）
        if save_dir:
            # 确保保存目录存在
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{metric_key}_curve.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"{metric_chinese}图表已保存到: {save_path}")

        # 关闭图表以释放内存
        plt.close()


# 修改plot_loss_curve函数，添加移动平均线
def plot_loss_curve(train_losses, eval_losses, save_dir=None):
    """
    绘制训练和测试损失变化曲线

    Args:
        train_losses: 训练损失历史记录
        eval_losses: 测试损失历史记录
        save_dir: 图片保存目录（可选）
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    # 计算移动平均线（窗口大小为3）
    window_size = 3
    train_ma = np.convolve(train_losses, np.ones(window_size) / window_size, mode='valid')
    eval_ma = np.convolve(eval_losses, np.ones(window_size) / window_size, mode='valid')

    # 创建新的epoch序列，与移动平均线长度匹配
    ma_epochs = range(window_size // 2 + 1, len(train_losses) - window_size // 2 + 1)

    # 绘制原始数据点
    plt.plot(epochs, train_losses, label='训练损失', marker='o', linewidth=1, alpha=0.7)
    plt.plot(epochs, eval_losses, label='测试损失', marker='s', linewidth=1, alpha=0.7)

    # 绘制移动平均线
    plt.plot(ma_epochs, train_ma, label='训练损失-移动平均', linewidth=2, color='blue')
    plt.plot(ma_epochs, eval_ma, label='测试损失-移动平均', linewidth=2, color='orange')

    plt.title('训练过程损失变化', fontproperties=get_chinese_font())
    plt.xlabel('Epoch', fontproperties=get_chinese_font())
    plt.ylabel('Loss', fontproperties=get_chinese_font())
    plt.legend(prop=get_chinese_font())
    plt.grid(True, alpha=0.3)

    # 保存图片（如果指定了保存路径）
    if save_dir:
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失图表已保存到: {save_path}")

    # 关闭图表以释放内存
    plt.close()


def get_chinese_font():
    """
    获取中文字体对象
    """
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
    available_fonts = {f.name: f for f in fm.fontManager.ttflist}

    for font_name in chinese_fonts:
        if font_name in available_fonts:
            return fm.FontProperties(fname=available_fonts[font_name].fname)

    # 如果没有找到合适的中文字体，返回默认字体
    return fm.FontProperties()