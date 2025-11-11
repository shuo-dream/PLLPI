import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# streamlit run 1demo.py

# 设置页面配置
st.set_page_config(
    page_title="PLLPI 演示系统",
    page_icon="🧬",
    layout="wide"
)


def main():
    st.title("🧬 PLLPI lncRNA-蛋白质相互作用预测演示")
    st.markdown("---")

    # 侧边栏
    st.sidebar.header("应用查看")

    # 选择演示功能
    demo_option = st.sidebar.selectbox(
        "选择演示功能",
        ["模型介绍", "物理损失介绍", "结果分析"]
    )

    if demo_option == "模型介绍":
        show_model_introduction()
    elif demo_option == "物理损失介绍":
        show_physics_loss_introduction()
    elif demo_option == "结果分析":
        show_result_analysis()



def show_model_introduction():
    st.header("PLLPI 模型介绍")

    st.subheader("🧬 研究背景")
    st.markdown("""
    ### 项目目的
    本项目旨在开发一种基于深度学习的预测模型，用于准确预测lncRNA（长链非编码RNA）与蛋白质之间的相互作用关系。
    通过识别这些相互作用，我们可以更好地理解基因调控机制、细胞功能以及相关疾病的发病机理，为生物医学研究
    和药物开发提供有力支持。
    """)

    # 显示模型结构图（示意）
    st.subheader("🏗️ 流程图")
    st.image("figure/1.png",
             caption="流程图", use_column_width=True)

    st.markdown("""
    ### 模型各阶段维度说明
    """)
    st.code("""
1. 原始输入数据：lncRNA和蛋白质的ID以及对应的序列
2. 初步提取特征：lncRNA具有3个物理特征（静电势、疏水性、堆积能），蛋白质具有4个物理特征（疏水性、体积、极性、净电荷）
3. 经过一维卷积层自适应平均池化层： lncrna_feature:(1097,128)          protein_feature(144,128)
4. 邻居节点信息聚合后特征维度：    lncrna:torch.Size([1097, 64])      protein:torch.Size([72, 64])
5. 深度特征提取后维度：           lncRNA: torch.Size([64, 128])      protein: torch.Size([64, 128])   (表示当前批次中有64个样本)
6. 交叉注意力后维度：            lncRNA: torch.Size([64, 128])      protein: torch.Size([64, 128])
7. 拼接后特征维度：              torch.Size([64, 256])
8. 交互建模层：                 torch.Size([64, 128])
9. 预测结果：                   torch.Size([64])
10. 最后输出：lncRNA与蛋白质之间是否存在相互作用的二分类预测结果（相互作用/无相互作用）以及相应的置信度得分（通过sigmoid函数将logits转换为概率值）
""")

    st.subheader("🤖 模型架构")
    st.markdown("""
    PLLPI模型采用以下技术构建，各部分作用和方法如下：
    """)
    st.markdown("""
<div style="white-space: pre;">3. <b>经过一维卷积层自适应平均池化层</b>： 使用1D卷积神经网络结合自适应平均池化处理变长序列，提取序列局部模式并统一特征维度
4. <b>邻居节点信息聚合后特征维度</b>： 通过图神经网络聚合邻居节点信息(使用gat进行信息聚合)
5. <b>深度特征提取后维度</b>： 使用使用1D卷积神经网络提取深度特征，包含8层卷积层和残差连接
6. <b>交叉注意力后维度</b>： 应用交叉注意力机制捕获lncRNA和蛋白质间的相互作用关系
7. <b>拼接后特征维度</b>： 在特征维度上拼接lncRNA和蛋白质特征形成联合表示
8. <b>交互建模层</b>： 通过全连接网络MLP(两个线性层+dropout+relu)学习复合特征表示
9. <b>预测结果</b>： 使用全连接网络(两个线性层+dropout+relu)结合Sigmoid函数输出相互作用概率
</div>
""", unsafe_allow_html=True)

    st.subheader("📈 模型性能")
    st.markdown("""
    该模型经过两次训练和优化，在测试集上表现良好。
    """)
    
    # 读取PLLPI.txt中的内容并解析显示
    try:
        with open("../../PLLPI/PLLPI.txt", "r", encoding="utf-8") as file:
            content = file.read()
        
        # 分割内容为两部分：默认参数和优化参数后
        parts = content.split("优化参数后的")
        default_params_content = parts[0].strip()
        optimized_params_content = parts[1].strip() if len(parts) > 1 else ""
        
        # 横向展示两部分内容
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 默认参数（未经过超参数优化）")
            st.text(default_params_content)
            
        with col2:
            st.markdown("#### 优化参数后（使用optuna进行超参数优化）")
            st.text(optimized_params_content)
            
    except FileNotFoundError:
        st.error("无法找到性能指标文件 PLLPI.txt")
    except Exception as e:
        st.error(f"读取性能指标文件时出错: {str(e)}")


def show_result_analysis():
    st.header("📈 结果分析")

    st.subheader("🏗️ 验证集平均指标和最好指标对比")
    # 第一行放两个图片
    col, col2 = st.columns(2)
    with col:
        st.image("./figure/val_model_comparison.png", caption="验证集平均指标", use_column_width=True)
    with col2:
        st.image("./figure/best_metrics_model_comparison.png", caption="最好指标对比", use_column_width=True)

    st.subheader("🏗️ AUC对比")
    # 第一行放两个图片
    col, col2 = st.columns(2)
    with col:
        st.image("./figure/auc.png", caption="PLLPI AUC", use_column_width=True)
    with col2:
        st.image("./figure/B_auc.png", caption="PLLPI_PL_A AUC", use_column_width=True)
    # 第二行放一个图片
    col2, col3, = st.columns(2)
    with col2:
        st.image("./figure/B_auc.png", caption="PLLPI_PL_B AUC", use_column_width=True)
    with col3:
        st.image("./figure/C_auc.png", caption="PLLPI_PL_C AUC", use_column_width=True)

    st.subheader("🏗️ loss对比")
    # 第一行放两个图片
    col, col2 = st.columns(2)
    with col:
        st.image("./figure/loss.png", caption="PLLPI loss", use_column_width=True)
    with col2:
        st.image("./figure/A_loss.png", caption="PLLPI_PL_A loss", use_column_width=True)
    # 第二行放一个图片
    col2, col3, = st.columns(2)
    with col2:
        st.image("./figure/B_loss.png", caption="PLLPI_PL_B loss", use_column_width=True)
    with col3:
        st.image("./figure/C_loss.png", caption="PLLPI_PL_C loss", use_column_width=True)


def show_physics_loss_introduction():
    st.header("⚛️ 物理损失介绍")
    
    st.subheader("物理损失的概念")
    st.markdown("""
    物理损失是一种将物理规律或先验知识融入深度学习模型训练过程的技术。在lncRNA-蛋白质相互作用预测任务中，
    我们利用物理化学特性（如疏水性）来构建物理一致性约束，使模型的预测结果不仅符合训练数据，也符合基本的物理规律。
    """)
    
    st.subheader("三种物理损失方法对比")
    st.markdown("""
    在项目中，我们探索了三种不同的物理损失实现方法，分别在PLLPI_PL_A、PLLPI_PL_B和PLLPI_PL_C目录中实现。
    """)
    
    # PLLPI_PL_A 方法
    st.markdown("#### 1. PLLPI_PL_A: 基于Embedding的物理损失")
    st.markdown("""
    - **核心思想**：基于模型输出的lncRNA和蛋白质embedding向量，通过可学习的head映射到物理特征空间，
      然后计算物理相似度矩阵，使其与主模型预测结果保持一致
    - **特征来源**：模型内部的embedding表示
    - **实现方式**：
      - 使用双线性变换或MLP将embedding映射到物理特征空间
      - 计算物理相似度矩阵: `S_k = sigmoid(L @ P^T)`
      - 通过MSE损失计算主模型预测结果与基于物理特征的预测结果之间的一致性
    """)
    
    st.code("""
# PLLPI_PL_A中的物理损失实现
class PhysicsLoss(nn.Module):
    def __init__(self, embedding_dim=128, num_physics_types=4):
        # 为每种物理量创建一个双线性head
        self.physics_heads = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in range(num_physics_types)
        ])
    
    def forward(self, lncrna_embeddings, protein_embeddings):
        # 通过head映射到物理特征空间
        lncrna_physics = head(lncrna_embeddings)
        protein_physics = head(protein_embeddings)
        # 计算相似度矩阵(使用矩阵乘法计算点积)
        similarity_matrix = torch.sigmoid(torch.matmul(lncrna_physics, protein_physics.transpose(-2, -1)))
""", language="python")
    
    # PLLPI_PL_B 方法
    st.markdown("#### 2. PLLPI_PL_B: 基于原始物理特征的直接物理损失")
    st.markdown("""
    - **核心思想**：直接使用从序列中提取的原始物理特征（如疏水性）计算物理相互作用矩阵，
      使主模型预测结果与基于物理规律的预测保持一致
    - **特征来源**：从氨基酸/核苷酸序列中提取的原始物理特征
    - **实现方式**：
      - 从序列中提取物理特征（如疏水性）
      - 使用外积运算计算物理相似度矩阵: `S_hydro = torch.outer(hydro_rna, hydro_protein)`
      - 通过MSE损失计算与主预测结果的一致性
    """)
    
    st.code("""
# PLLPI_PL_B中的物理损失实现
def compute_physical_matrices(self, lncrna_physics, protein_physics):
    # 提取疏水性特征
    hydro_rna = lncrna_physics[:, 0]
    hydro_protein = protein_physics[:, 0]
    # 计算疏水性相互作用矩阵
    S_hydro = torch.outer(hydro_rna, hydro_protein)
    S_hydro = torch.sigmoid(S_hydro)
""", language="python")
    
    # PLLPI_PL_C 方法
    st.markdown("#### 3. PLLPI_PL_C: 混合物理损失")
    st.markdown("""
    - **核心思想**：结合前两种方法的优势，同时使用embedding和原始物理特征计算物理相似度矩阵，
      通过加权融合得到最终的物理一致性约束
    - **特征来源**：模型embedding和原始物理特征的组合
    - **实现方式**：
      - 分别基于embedding和原始物理特征计算相似度矩阵
      - 通过加权融合两种相似度矩阵: `S_combined = α*S_emb + (1-α)*S_raw`
      - 通过MSE损失计算与主预测结果的一致性
    """)
    
    st.code("""
# PLLPI_PL_C中的混合物理损失实现
def forward(self, lncrna_embeddings, protein_embeddings, lncrna_physics, protein_physics):
    # 分别计算基于embedding和原始物理特征的相似度矩阵
    embedding_similarity_matrices = self._compute_similarity_from_embeddings(lncrna_embeddings, protein_embeddings)
    raw_similarity_matrices = self._compute_similarity_from_physics(lncrna_physics, protein_physics)
    # 融合两种相似度矩阵
    combined_matrix = self.alpha * embedding_similarity_matrices[i] + (1 - self.alpha) * raw_similarity_matrices[i]
""", language="python")
    
    st.subheader("三种方法的比较")
    
    comparison_data = pd.DataFrame({
        "方法": ["PLLPI_PL_A", "PLLPI_PL_B", "PLLPI_PL_C"],
        "特征来源": ["模型Embedding", "原始物理特征", "Embedding+原始特征"],
        "实现复杂度": ["中等", "简单", "复杂"],
        "可解释性": ["中等", "高", "高"],
        "计算开销": ["中等", "低", "高"],
        "物理一致性": ["中等", "高", "最高"]
    })
    
    st.table(comparison_data)

if __name__ == "__main__":
    main()