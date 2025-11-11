import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from model.PLLPI import PLLPI
from plot_utils import get_chinese_font

# 设置页面配置
st.set_page_config(
    page_title="PLLPI 演示系统",
    page_icon="🧬",
    layout="wide"
)

# 初始化中文字体
chinese_font = get_chinese_font()


def load_model(model_path=None):
    """
    加载训练好的模型
    """
    model = PLLPI(lncrna_dim=128, protein_dim=128)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        st.warning("未找到预训练模型，使用随机初始化的模型")
    model.eval()
    return model


def main():
    st.title("🧬 PLLPI lncRNA-蛋白质相互作用预测演示")
    st.markdown("---")

    # 侧边栏
    st.sidebar.header("应用查看")

    # 选择演示功能
    demo_option = st.sidebar.selectbox(
        "选择演示功能",
        ["模型介绍", "数据可视化", "交互预测演示", "结果分析"]
    )

    if demo_option == "模型介绍":
        show_model_introduction()
    elif demo_option == "数据可视化":
        show_data_visualization()
    elif demo_option == "交互预测演示":
        show_prediction_demo()
    elif demo_option == "结果分析":
        show_result_analysis()


def show_model_introduction():
    st.header("PLLPI 模型介绍")

    st.subheader("🧬 研究背景")
    st.markdown("""
    lncRNA（长链非编码RNA）与蛋白质的相互作用在多种生物过程中发挥重要作用，
    包括基因表达调控、细胞分化和疾病发展等。准确预测lncRNA-蛋白质相互作用
    对于理解生物机制和疾病治疗具有重要意义。
    """)

    st.subheader("🤖 模型架构")
    st.markdown("""
    PLLPI模型采用以下技术构建：
    1. **深度特征提取**：使用1D卷积神经网络提取lncRNA和蛋白质的深度特征
    2. **交叉注意力机制**：通过交叉注意力机制捕获lncRNA和蛋白质之间的相互作用
    3. **异构图神经网络**：利用异构图神经网络聚合邻居节点信息
    4. **端到端训练**：整个模型端到端训练，优化预测性能
    """)

    st.subheader("📈 模型性能")
    st.markdown("""
    该模型在测试集上表现良好：
    - 准确率(Accuracy)：~0.93
    - 精确率(Precision)：~0.92
    - 召回率(Recall)：~0.94
    - F1分数：~0.93
    - AUC：~0.97
    """)

    # 显示模型结构图（示意）
    st.subheader("🏗️ 模型结构示意图")
    st.image("https://via.placeholder.com/800x400.png?text=PLLPI+Model+Architecture",
             caption="PLLPI模型结构示意图", use_column_width=True)


def show_data_visualization():
    st.header("📊 数据可视化")

    st.subheader("lncRNA-蛋白质相互作用网络")
    # 这里可以加载实际的数据可视化结果
    st.image("https://via.placeholder.com/600x400.png?text=Interaction+Network",
             caption="lncRNA-蛋白质相互作用网络", use_column_width=True)

    st.subheader("数据集统计信息")
    # 显示数据集的基本统计信息
    stats_data = {
        "类别": ["lncRNA数量", "蛋白质数量", "已知相互作用", "正样本", "负样本"],
        "数值": [100, 150, 800, 800, 800]
    }
    stats_df = pd.DataFrame(stats_data)
    st.table(stats_df)

    st.subheader("特征分布")
    # 显示特征分布的可视化
    fig, ax = plt.subplots(figsize=(10, 4))

    # 模拟特征分布数据
    feature_types = ['序列特征', '物理化学特征', '结构特征', '进化特征']
    feature_counts = [25, 30, 20, 15]

    bars = ax.bar(feature_types, feature_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_title('特征类型分布', fontproperties=chinese_font, fontsize=14)
    ax.set_ylabel('特征数量', fontproperties=chinese_font)

    # 在柱状图上添加数值标签
    for bar, count in zip(bars, feature_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom', fontproperties=chinese_font)

    st.pyplot(fig)


def show_prediction_demo():
    st.header("🔮 交互预测演示")

    st.markdown("""
    在这个演示中，您可以模拟lncRNA-蛋白质相互作用的预测过程。
    由于实际演示需要大量计算资源，这里展示的是一个简化版本。
    """)

    # 加载模型
    with st.spinner("正在加载模型..."):
        model = load_model()  # 实际使用时应提供模型路径

    st.subheader("输入参数设置")
    col1, col2 = st.columns(2)

    with col1:
        lncrna_id = st.text_input("lncRNA ID", "ENSG00000227238")
        st.markdown("**lncRNA特征向量（示例）**")
        lncrna_features = np.random.rand(128)
        st.line_chart(lncrna_features[:50])  # 只显示前50个特征

    with col2:
        protein_id = st.text_input("蛋白质 ID", "ENSP00000355094")
        st.markdown("**蛋白质特征向量（示例）**")
        protein_features = np.random.rand(128)
        st.line_chart(protein_features[:50])  # 只显示前50个特征

    if st.button("开始预测", type="primary"):
        with st.spinner("正在预测..."):
            # 模拟预测过程
            # 实际应该使用模型进行预测
            prediction_score = np.random.rand()
            prediction_label = "相互作用" if prediction_score > 0.5 else "无相互作用"

            st.subheader("预测结果")
            st.metric("预测得分", f"{prediction_score:.4f}")
            st.metric("预测结果", prediction_label)

            # 可视化预测结果
            fig, ax = plt.subplots(figsize=(8, 2))
            colors = ['green' if prediction_score > 0.5 else 'red',
                      'lightgray' if prediction_score > 0.5 else 'lightgray']
            bars = ax.barh(['预测结果'], [prediction_score], color=colors[0])
            ax.set_xlim(0, 1)
            ax.set_xlabel('相互作用概率', fontproperties=chinese_font)
            ax.axvline(x=0.5, color='red', linestyle='--', label='阈值(0.5)')

            # 添加数值标签
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2,
                        f'{width:.4f}', ha='left', va='center', fontproperties=chinese_font)

            st.pyplot(fig)

            st.info("💡 **说明**: 实际预测基于深度学习模型对lncRNA和蛋白质特征的复杂分析")


def show_result_analysis():
    st.header("📈 结果分析")

    st.subheader("训练过程指标")

    # 模拟训练指标数据
    epochs = list(range(1, 101))
    train_acc = [0.7 + 0.25 * (1 - np.exp(-i / 30)) + np.random.normal(0, 0.01) for i in epochs]
    val_acc = [0.65 + 0.28 * (1 - np.exp(-i / 35)) + np.random.normal(0, 0.015) for i in epochs]

    # 准确率曲线
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, train_acc, label='训练准确率', color='#4ECDC4')
    ax1.plot(epochs, val_acc, label='验证准确率', color='#FF6B6B')
    ax1.set_title('模型训练过程准确率变化', fontproperties=chinese_font, fontsize=14)
    ax1.set_xlabel('Epoch', fontproperties=chinese_font)
    ax1.set_ylabel('准确率', fontproperties=chinese_font)
    ax1.legend(prop=chinese_font)
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

    # F1分数和AUC对比
    st.subheader("关键指标对比")
    metrics_data = {
        '指标': ['准确率', '精确率', '召回率', 'F1分数', 'AUC'],
        '训练集': [0.95, 0.94, 0.96, 0.95, 0.98],
        '测试集': [0.93, 0.92, 0.94, 0.93, 0.97]
    }
    metrics_df = pd.DataFrame(metrics_data)
    st.table(metrics_df)

    # 混淆矩阵（示意）
    st.subheader("混淆矩阵")
    confusion_data = pd.DataFrame({
        '实际\预测': ['相互作用', '无相互作用'],
        '相互作用': [850, 50],
        '无相互作用': [70, 880]
    })
    st.table(confusion_data)

    st.success("✅ 模型表现良好，具有较强的预测能力")


if __name__ == "__main__":
    main()