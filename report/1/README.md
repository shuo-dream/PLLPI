# PLLPI 演示程序

这是一个用于演示 PLLPI (Predicting lncRNA-Protein Interactions) 项目的简化版本程序。

## 项目介绍

PLLPI 是一个基于深度学习的模型，用于预测长链非编码RNA (lncRNA) 与蛋白质之间的相互作用。该模型结合了以下技术：

1. **深度特征提取**：使用1D卷积神经网络提取lncRNA和蛋白质的深度特征
2. **交叉注意力机制**：通过交叉注意力机制捕获lncRNA和蛋白质之间的相互作用
3. **异构图神经网络**：利用异构图神经网络聚合邻居节点信息
4. **端到端训练**：整个模型端到端训练，优化预测性能

## 模型性能

该模型在测试集上表现良好：
- 准确率(Accuracy)：~0.93
- 精确率(Precision)：~0.92
- 召回率(Recall)：~0.94
- F1分数：~0.93
- AUC：~0.97

## 演示程序功能

本演示程序包含以下功能模块：

1. **模型介绍**：介绍PLLPI模型的背景、架构和性能
2. **数据可视化**：展示lncRNA-蛋白质相互作用网络和数据集统计信息
3. **交互预测演示**：模拟lncRNA-蛋白质相互作用的预测过程
4. **结果分析**：展示模型训练过程中的指标变化

## 运行环境

- Python 3.7+
- Streamlit
- PyTorch
- Pandas
- NumPy
- Matplotlib

## 运行方法

1. 安装依赖：
   ```
   pip install streamlit==1.29.0 altair==4.2.2 torch pandas numpy matplotlib
   ```

2. 运行演示程序：
   ```
   streamlit run demo_app.py
   ```

## 文件说明

- `demo_app.py`: 演示程序主文件
- `README.md`: 本说明文件

## 注意事项

由于演示程序运行在资源受限的环境中，实际预测功能使用了模拟数据而非真实模型预测。