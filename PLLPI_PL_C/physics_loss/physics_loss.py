import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLossCombined(nn.Module):
    """
    结合embedding和原始物理特征的物理损失模块（方案C）
    只使用疏水性特征
    """

    def __init__(self, embedding_dim=128, physics_feature_dim=1, combined_dim=129, num_physics_types=1, alpha=0.5):
        """
        初始化混合物理损失模块

        Args:
            embedding_dim (int): embedding维度
            physics_feature_dim (int): 原始物理特征维度（仅疏水性）
            combined_dim (int): 拼接后的特征维度 (embedding_dim + physics_feature_dim)
            num_physics_types (int): 物理类型数量（仅疏水性）
            alpha (float): 融合权重，S_k = α*S_emb_k + (1-α)*S_raw_k
        """
        super(PhysicsLossCombined, self).__init__()
        self.embedding_dim = embedding_dim
        self.physics_feature_dim = physics_feature_dim
        self.combined_dim = combined_dim
        self.num_physics_types = num_physics_types  # 这里应该是1，因为只使用疏水性特征
        self.alpha = alpha  # 融合权重

        # 为每种物理量创建一个双线性head，处理拼接后的特征
        self.physics_heads = nn.ModuleList([
            nn.Linear(embedding_dim, combined_dim, bias=False)
            for _ in range(num_physics_types)
        ])

        # 物理类型名称
        self.physics_types = ['hydrophobicity']

    def forward(self, lncrna_embeddings, protein_embeddings, lncrna_physics, protein_physics):
        """
        前向传播，计算物理相似度矩阵

        Args:
            lncrna_embeddings (torch.Tensor): lncRNA的embedding，形状为[batch_size, embedding_dim]
            protein_embeddings (torch.Tensor): 蛋白质的embedding，形状为[batch_size, embedding_dim]
            lncrna_physics (torch.Tensor): lncRNA的原始物理特征，形状为[batch_size, physics_feature_dim]
            protein_physics (torch.Tensor): 蛋白质的原始物理特征，形状为[batch_size, physics_feature_dim]

        Returns:
            tuple: (combined_similarity_matrices, embedding_similarity_matrices, raw_similarity_matrices)
                每个都是物理相似度矩阵列表，每个矩阵形状为[batch_size, batch_size]
        """
        # 拼接embedding和原始物理特征
        lncrna_combined = torch.cat([lncrna_embeddings, lncrna_physics], dim=1)  # [batch_size, combined_dim]
        protein_combined = torch.cat([protein_embeddings, protein_physics], dim=1)  # [batch_size, combined_dim]

        # 分别计算基于embedding和原始物理特征的相似度矩阵
        embedding_similarity_matrices = self._compute_similarity_from_embeddings(lncrna_embeddings, protein_embeddings)
        raw_similarity_matrices = self._compute_similarity_from_physics(lncrna_physics, protein_physics)

        # 融合两种相似度矩阵
        combined_similarity_matrices = []
        for i in range(self.num_physics_types):
            combined_matrix = self.alpha * embedding_similarity_matrices[i] + (1 - self.alpha) * \
                              raw_similarity_matrices[i]
            combined_similarity_matrices.append(combined_matrix)

        return combined_similarity_matrices, embedding_similarity_matrices, raw_similarity_matrices

    def _compute_similarity_from_embeddings(self, lncrna_embeddings, protein_embeddings):
        """
        基于embedding计算物理相似度矩阵（类似方案A）

        Args:
            lncrna_embeddings (torch.Tensor): lncRNA的embedding
            protein_embeddings (torch.Tensor): 蛋白质的embedding

        Returns:
            list: 物理相似度矩阵列表
        """
        similarity_matrices = []

        # 对每种物理类型计算相似度矩阵
        for head in self.physics_heads:
            # 通过head映射到物理特征空间
            lncrna_physics = head(lncrna_embeddings)  # [batch_size, combined_dim]
            protein_physics = head(protein_embeddings)  # [batch_size, combined_dim]

            # 计算相似度矩阵 S_k = sigmoid(L @ P^T)
            similarity_matrix = torch.sigmoid(torch.matmul(lncrna_physics, protein_physics.transpose(-2, -1)))
            similarity_matrices.append(similarity_matrix)

        return similarity_matrices

    def _compute_similarity_from_physics(self, lncrna_physics, protein_physics):
        """
        基于原始物理特征计算物理相互作用矩阵（类似方案B）

        Args:
            lncrna_physics (torch.Tensor): lncRNA的原始物理特征
            protein_physics (torch.Tensor): 蛋白质的原始物理特征

        Returns:
            list: 物理相似度矩阵列表
        """
        similarity_matrices = []

        # 只使用疏水性特征（索引0）
        if lncrna_physics.shape[1] > 0 and protein_physics.shape[1] > 0:
            # 获取疏水性特征
            lncrna_feature = lncrna_physics[:, 0]  # [batch_size]
            protein_feature = protein_physics[:, 0]  # [batch_size]

            # 计算外积相似度
            similarity_matrix = torch.sigmoid(torch.outer(lncrna_feature, protein_feature))
            similarity_matrices.append(similarity_matrix)

        return similarity_matrices

    def compute_physics_loss(self, predicted_interaction, physics_similarity_matrices, weights=None):
        """
        计算物理一致性损失

        Args:
            predicted_interaction (torch.Tensor): 主模型预测的交互概率矩阵，形状为[batch_size, batch_size]
            physics_similarity_matrices (list): 物理相似度矩阵列表
            weights (list, optional): 每种物理损失的权重

        Returns:
            torch.Tensor: 总的物理一致性损失
        """
        if weights is None:
            weights = [1.0] * self.num_physics_types

        total_physics_loss = 0.0

        # 计算每种物理类型与主预测的一致性损失
        for i, similarity_matrix in enumerate(physics_similarity_matrices):
            # 确保两个矩阵的尺寸匹配
            # 检查predicted_interaction是否为1D张量，如果是则将其转换为2D
            if predicted_interaction.dim() == 1:
                predicted_interaction = predicted_interaction.unsqueeze(0)

            if predicted_interaction.shape != similarity_matrix.shape:
                # 进行形状调整
                min_rows = min(predicted_interaction.shape[0], similarity_matrix.shape[0])
                min_cols = min(predicted_interaction.shape[1], similarity_matrix.shape[1])
                predicted_interaction_adj = predicted_interaction[:min_rows, :min_cols]
                similarity_matrix_adj = similarity_matrix[:min_rows, :min_cols]
            else:
                predicted_interaction_adj = predicted_interaction
                similarity_matrix_adj = similarity_matrix

            # 使用MSE损失计算一致性
            '''
                predicted_interaction_adj: 主模型的预测交互概率矩阵（经过形状调整后）
                similarity_matrix_adj: 基于物理特征计算出的相似度矩阵（经过形状调整后）
                F.mse_loss: PyTorch内置的均方误差损失函数
            '''
            physics_loss = F.mse_loss(predicted_interaction_adj, similarity_matrix_adj)
            total_physics_loss += weights[i] * physics_loss

        return total_physics_loss


class PhysicsLossCombinedEnhanced(nn.Module):
    """
    增强版混合物理损失模块，使用MLP而不是简单的线性变换
    只使用疏水性特征
    """

    def __init__(self, embedding_dim=128, physics_feature_dim=1, combined_dim=129, hidden_dim=64, num_physics_types=1,
                 alpha=0.5):
        """
        初始化增强版混合物理损失模块

        Args:
            embedding_dim (int): embedding维度
            physics_feature_dim (int): 原始物理特征维度（仅疏水性）
            combined_dim (int): 拼接后的特征维度
            hidden_dim (int): MLP隐藏层维度
            num_physics_types (int): 物理类型数量（仅疏水性）
            alpha (float): 融合权重
        """
        super(PhysicsLossCombinedEnhanced, self).__init__()
        self.embedding_dim = embedding_dim
        self.physics_feature_dim = physics_feature_dim
        self.combined_dim = combined_dim
        self.hidden_dim = hidden_dim
        self.num_physics_types = num_physics_types
        self.alpha = alpha

        # 为每种物理量创建一个MLP head
        self.physics_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, combined_dim)
            )
            for _ in range(num_physics_types)
        ])

        # 物理类型名称
        self.physics_types = ['hydrophobicity']

    def forward(self, lncrna_embeddings, protein_embeddings, lncrna_physics, protein_physics):
        """
        前向传播，计算物理相似度矩阵

        Args:
            lncrna_embeddings (torch.Tensor): lncRNA的embedding，形状为[batch_size, embedding_dim]
            protein_embeddings (torch.Tensor): 蛋白质的embedding，形状为[batch_size, embedding_dim]
            lncrna_physics (torch.Tensor): lncRNA的原始物理特征，形状为[batch_size, physics_feature_dim]
            protein_physics (torch.Tensor): 蛋白质的原始物理特征，形状为[batch_size, physics_feature_dim]

        Returns:
            tuple: (combined_similarity_matrices, embedding_similarity_matrices, raw_similarity_matrices)
        """
        # 拼接embedding和原始物理特征
        lncrna_combined = torch.cat([lncrna_embeddings, lncrna_physics], dim=1)  # [batch_size, combined_dim]
        protein_combined = torch.cat([protein_embeddings, protein_physics], dim=1)  # [batch_size, combined_dim]

        # 分别计算基于embedding和原始物理特征的相似度矩阵
        embedding_similarity_matrices = self._compute_similarity_from_embeddings(lncrna_embeddings, protein_embeddings)
        raw_similarity_matrices = self._compute_similarity_from_physics(lncrna_physics, protein_physics)

        # 融合两种相似度矩阵
        combined_similarity_matrices = []
        for i in range(self.num_physics_types):
            combined_matrix = self.alpha * embedding_similarity_matrices[i] + (1 - self.alpha) * \
                              raw_similarity_matrices[i]
            combined_similarity_matrices.append(combined_matrix)

        return combined_similarity_matrices, embedding_similarity_matrices, raw_similarity_matrices

    def _compute_similarity_from_embeddings(self, lncrna_embeddings, protein_embeddings):
        """
        基于embedding计算物理相似度矩阵

        Args:
            lncrna_embeddings (torch.Tensor): lncRNA的embedding
            protein_embeddings (torch.Tensor): 蛋白质的embedding

        Returns:
            list: 物理相似度矩阵列表
        """
        similarity_matrices = []

        # 对每种物理类型计算相似度矩阵
        for head in self.physics_heads:
            # 通过MLP head映射到物理特征空间
            lncrna_physics = head(lncrna_embeddings)  # [batch_size, combined_dim]
            protein_physics = head(protein_embeddings)  # [batch_size, combined_dim]

            # 计算相似度矩阵 S_k = sigmoid(L @ P^T)
            similarity_matrix = torch.sigmoid(torch.matmul(lncrna_physics, protein_physics.transpose(-2, -1)))
            similarity_matrices.append(similarity_matrix)

        return similarity_matrices

    def _compute_similarity_from_physics(self, lncrna_physics, protein_physics):
        """
        基于原始物理特征计算物理相互作用矩阵

        Args:
            lncrna_physics (torch.Tensor): lncRNA的原始物理特征
            protein_physics (torch.Tensor): 蛋白质的原始物理特征

        Returns:
            list: 物理相似度矩阵列表
        """
        similarity_matrices = []

        # 只使用疏水性特征（索引0）
        if lncrna_physics.shape[1] > 0 and protein_physics.shape[1] > 0:
            # 获取疏水性特征
            lncrna_feature = lncrna_physics[:, 0]  # [batch_size]
            protein_feature = protein_physics[:, 0]  # [batch_size]

            # 计算外积相似度
            similarity_matrix = torch.sigmoid(torch.outer(lncrna_feature, protein_feature))
            similarity_matrices.append(similarity_matrix)

        return similarity_matrices

    def compute_physics_loss(self, predicted_interaction, physics_similarity_matrices, weights=None):
        """
        计算物理一致性损失

        Args:
            predicted_interaction (torch.Tensor): 主模型预测的交互概率矩阵，形状为[batch_size, batch_size]
            physics_similarity_matrices (list): 物理相似度矩阵列表
            weights (list, optional): 每种物理损失的权重

        Returns:
            torch.Tensor: 总的物理一致性损失
        """
        if weights is None:
            weights = [1.0] * self.num_physics_types

        total_physics_loss = 0.0

        # 计算每种物理类型与主预测的一致性损失
        for i, similarity_matrix in enumerate(physics_similarity_matrices):
            # 确保两个矩阵的尺寸匹配
            # 检查predicted_interaction是否为1D张量，如果是则将其转换为2D
            if predicted_interaction.dim() == 1:
                predicted_interaction = predicted_interaction.unsqueeze(0)

            if predicted_interaction.shape != similarity_matrix.shape:
                # 进行形状调整
                min_rows = min(predicted_interaction.shape[0], similarity_matrix.shape[0])
                min_cols = min(predicted_interaction.shape[1], similarity_matrix.shape[1])
                predicted_interaction_adj = predicted_interaction[:min_rows, :min_cols]
                similarity_matrix_adj = similarity_matrix[:min_rows, :min_cols]
            else:
                predicted_interaction_adj = predicted_interaction
                similarity_matrix_adj = similarity_matrix

            # 使用MSE损失计算一致性
            physics_loss = F.mse_loss(predicted_interaction_adj, similarity_matrix_adj)
            total_physics_loss += weights[i] * physics_loss

        return total_physics_loss


def test_physics_loss_combined():
    """
    测试混合物理损失模块
    """
    # 创建测试数据
    batch_size = 32
    embedding_dim = 128
    physics_feature_dim = 1  # 只使用疏水性特征

    lncrna_embeddings = torch.randn(batch_size, embedding_dim)
    protein_embeddings = torch.randn(batch_size, embedding_dim)
    lncrna_physics = torch.randn(batch_size, physics_feature_dim)
    protein_physics = torch.randn(batch_size, physics_feature_dim)
    predicted_interaction = torch.sigmoid(torch.randn(batch_size, batch_size))

    # 测试基础版本
    physics_loss_combined = PhysicsLossCombined(
        embedding_dim=embedding_dim,
        physics_feature_dim=physics_feature_dim,
        combined_dim=embedding_dim + physics_feature_dim,
        num_physics_types=1
    )

    combined_matrices, embedding_matrices, raw_matrices = physics_loss_combined(
        lncrna_embeddings, protein_embeddings, lncrna_physics, protein_physics
    )

    total_loss = physics_loss_combined.compute_physics_loss(predicted_interaction, combined_matrices)

    print("混合版本测试:")
    print(f"lncRNA embedding形状: {lncrna_embeddings.shape}")
    print(f"蛋白质embedding形状: {protein_embeddings.shape}")
    print(f"lncRNA物理特征形状: {lncrna_physics.shape}")
    print(f"蛋白质物理特征形状: {protein_physics.shape}")
    print(f"预测交互矩阵形状: {predicted_interaction.shape}")
    print(f"融合相似度矩阵数量: {len(combined_matrices)}")
    print(f"每个融合相似度矩阵形状: {combined_matrices[0].shape}")
    print(f"总的物理一致性损失: {total_loss.item()}")

    # 测试增强版本
    physics_loss_combined_enhanced = PhysicsLossCombinedEnhanced(
        embedding_dim=embedding_dim,
        physics_feature_dim=physics_feature_dim,
        combined_dim=embedding_dim + physics_feature_dim,
        hidden_dim=64,
        num_physics_types=1
    )

    combined_matrices_enhanced, _, _ = physics_loss_combined_enhanced(
        lncrna_embeddings, protein_embeddings, lncrna_physics, protein_physics
    )

    total_loss_enhanced = physics_loss_combined_enhanced.compute_physics_loss(predicted_interaction,
                                                                              combined_matrices_enhanced)

    print("\n增强混合版本测试:")
    print(f"融合相似度矩阵数量: {len(combined_matrices_enhanced)}")
    print(f"每个融合相似度矩阵形状: {combined_matrices_enhanced[0].shape}")
    print(f"总的物理一致性损失: {total_loss_enhanced.item()}")


if __name__ == "__main__":
    test_physics_loss_combined()