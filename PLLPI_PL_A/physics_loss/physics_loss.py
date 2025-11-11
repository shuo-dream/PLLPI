import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLoss(nn.Module):
    """
    物理损失模块，基于交叉注意力后的lncRNA和蛋白质embedding计算物理相似度
    """

    def __init__(self, embedding_dim=128, num_physics_types=4):
        """
        初始化物理损失模块

        Args:
            embedding_dim (int): embedding维度
            num_physics_types (int): 物理类型数量（疏水性、电荷、氢键、范德华等）
        """
        super(PhysicsLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_physics_types = num_physics_types

        # 为每种物理量创建一个双线性head
        # 每个head将embedding映射到物理特征空间
        self.physics_heads = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim, bias=False)
            for _ in range(num_physics_types)
        ])

        # 物理类型名称（仅供参考）
        self.physics_types = ['hydrophobicity', 'charge', 'hydrogen_bond', 'van_der_waals']

    def forward(self, lncrna_embeddings, protein_embeddings):
        """
        前向传播，计算物理相似度矩阵

        Args:
            lncrna_embeddings (torch.Tensor): lncRNA的embedding，形状为[batch_size, embedding_dim]
            protein_embeddings (torch.Tensor): 蛋白质的embedding，形状为[batch_size, embedding_dim]

        Returns:
            list: 每种物理类型的相似度矩阵列表，每个矩阵形状为[batch_size, batch_size]
        """
        physics_similarity_matrices = []

        # 对每种物理类型计算相似度矩阵
        for head in self.physics_heads:
            # 通过head映射到物理特征空间
            lncrna_physics = head(lncrna_embeddings)  # [batch_size, embedding_dim]
            protein_physics = head(protein_embeddings)  # [batch_size, embedding_dim]

            # 计算相似度矩阵 S_k = sigmoid(L @ P^T)
            # 这里使用点积相似度
            similarity_matrix = torch.sigmoid(torch.matmul(lncrna_physics, protein_physics.transpose(-2, -1)))
            physics_similarity_matrices.append(similarity_matrix)

        return physics_similarity_matrices

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
            # 使用MSE损失计算一致性
            physics_loss = F.mse_loss(predicted_interaction, similarity_matrix)
            total_physics_loss += weights[i] * physics_loss

        return total_physics_loss


class PhysicsLossEnhanced(nn.Module):
    """
    增强版物理损失模块，使用MLP而不是简单的线性变换
    """

    def __init__(self, embedding_dim=128, hidden_dim=64, num_physics_types=4):
        """
        初始化增强版物理损失模块

        Args:
            embedding_dim (int): embedding维度
            hidden_dim (int): MLP隐藏层维度
            num_physics_types (int): 物理类型数量
        """
        super(PhysicsLossEnhanced, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_physics_types = num_physics_types

        # 为每种物理量创建一个MLP head
        self.physics_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
            for _ in range(num_physics_types)
        ])

        # 物理类型名称（仅供参考）
        self.physics_types = ['hydrophobicity', 'charge', 'hydrogen_bond', 'van_der_waals']

    def forward(self, lncrna_embeddings, protein_embeddings):
        """
        前向传播，计算物理相似度矩阵

        Args:
            lncrna_embeddings (torch.Tensor): lncRNA的embedding，形状为[batch_size, embedding_dim]
            protein_embeddings (torch.Tensor): 蛋白质的embedding，形状为[batch_size, embedding_dim]

        Returns:
            list: 每种物理类型的相似度矩阵列表，每个矩阵形状为[batch_size, batch_size]
        """
        physics_similarity_matrices = []

        # 对每种物理类型计算相似度矩阵
        for head in self.physics_heads:
            # 通过MLP head映射到物理特征空间
            lncrna_physics = head(lncrna_embeddings)  # [batch_size, embedding_dim]
            protein_physics = head(protein_embeddings)  # [batch_size, embedding_dim]

            # 计算相似度矩阵 S_k = sigmoid(L @ P^T)
            similarity_matrix = torch.sigmoid(torch.matmul(lncrna_physics, protein_physics.transpose(-2, -1)))
            physics_similarity_matrices.append(similarity_matrix)

        return physics_similarity_matrices

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
            # 使用MSE损失计算一致性
            physics_loss = F.mse_loss(predicted_interaction, similarity_matrix)
            total_physics_loss += weights[i] * physics_loss

        return total_physics_loss


def test_physics_loss():
    """
    测试物理损失模块
    """
    # 创建测试数据
    batch_size = 32
    embedding_dim = 128

    lncrna_embeddings = torch.randn(batch_size, embedding_dim)
    protein_embeddings = torch.randn(batch_size, embedding_dim)
    predicted_interaction = torch.sigmoid(torch.randn(batch_size, batch_size))

    # 测试基础版本
    physics_loss_module = PhysicsLoss(embedding_dim=embedding_dim, num_physics_types=4)
    similarity_matrices = physics_loss_module(lncrna_embeddings, protein_embeddings)
    total_loss = physics_loss_module.compute_physics_loss(predicted_interaction, similarity_matrices)

    print("基础版本测试:")
    print(f"lncRNA embedding形状: {lncrna_embeddings.shape}")
    print(f"蛋白质embedding形状: {protein_embeddings.shape}")
    print(f"预测交互矩阵形状: {predicted_interaction.shape}")
    print(f"相似度矩阵数量: {len(similarity_matrices)}")
    print(f"每个相似度矩阵形状: {similarity_matrices[0].shape}")
    print(f"总的物理一致性损失: {total_loss.item()}")

    # 测试增强版本
    physics_loss_enhanced = PhysicsLossEnhanced(embedding_dim=embedding_dim, hidden_dim=64, num_physics_types=4)
    similarity_matrices_enhanced = physics_loss_enhanced(lncrna_embeddings, protein_embeddings)
    total_loss_enhanced = physics_loss_enhanced.compute_physics_loss(predicted_interaction, similarity_matrices_enhanced)

    print("\n增强版本测试:")
    print(f"相似度矩阵数量: {len(similarity_matrices_enhanced)}")
    print(f"每个相似度矩阵形状: {similarity_matrices_enhanced[0].shape}")
    print(f"总的物理一致性损失: {total_loss_enhanced.item()}")


if __name__ == "__main__":
    test_physics_loss()