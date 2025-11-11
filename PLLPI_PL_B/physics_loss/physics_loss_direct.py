import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsLossDirect(nn.Module):
    """
    直接基于原始物理标量特征计算物理相互作用矩阵的物理损失模块
    不使用任何embedding，完全基于物理规则计算
    """

    def __init__(self):
        """
        初始化物理损失模块
        """
        super(PhysicsLossDirect, self).__init__()

        # 物理类型名称
        # 只使用疏水性特征
        self.physics_types = ['hydrophobicity']

        # 定义特征索引到名称的映射
        self.feature_names = {
            0: "疏水性 (hydrophobicity)"
        }

    def compute_physical_matrices(self, lncrna_physics, protein_physics):
        """
        基于原始物理标量特征计算各类物理相互作用矩阵

        Args:
            lncrna_physics (torch.Tensor): lncRNA的物理特征，形状为[batch_size, num_features_lncrna]
            protein_physics (torch.Tensor): 蛋白质的物理特征，形状为[batch_size, num_features_protein]

        Returns:
            dict: 每种物理类型的相互作用矩阵字典
        """
        # 只使用疏水性特征（索引0）
        physical_matrices = {}

        # 检查输入是否包含疏水性特征
        if lncrna_physics.shape[1] > 0 and protein_physics.shape[1] > 0:
            # 提取疏水性特征
            hydro_rna = lncrna_physics[:, 0]  # 第一个特征是疏水性
            hydro_protein = protein_physics[:, 0]  # 第一个特征是疏水性

            # 计算疏水性相互作用矩阵 S_hydro = outer(hydro_rna, hydro_protein)
            S_hydro = torch.outer(hydro_rna, hydro_protein)
            # 使用sigmoid确保值在[0,1]范围内
            S_hydro = torch.sigmoid(S_hydro)
            physical_matrices['hydrophobicity'] = S_hydro

        return physical_matrices

    def compute_physics_loss(self, predicted_interaction, physical_matrices, weights=None):
        """
        计算物理一致性损失

        Args:
            predicted_interaction (torch.Tensor): 主模型预测的交互概率矩阵，形状为[batch_size, batch_size]
            physical_matrices (dict): 物理相互作用矩阵字典
            weights (dict, optional): 每种物理损失的权重

        Returns:
            torch.Tensor: 总的物理一致性损失
        """
        if weights is None:
            weights = {key: 1.0 for key in self.physics_types}

        total_physics_loss = 0.0

        # 计算每种物理类型与主预测的一致性损失
        for physics_type, similarity_matrix in physical_matrices.items():
            # 确保两个矩阵的尺寸匹配
            if predicted_interaction.shape != similarity_matrix.shape:
                # 如果形状不匹配，进行插值或裁剪
                if predicted_interaction.shape[0] < similarity_matrix.shape[0]:
                    similarity_matrix = similarity_matrix[:predicted_interaction.shape[0],
                                        :predicted_interaction.shape[1]]
                elif predicted_interaction.shape[0] > similarity_matrix.shape[0]:
                    # 通过重复最后一行/列进行填充
                    pad_row = similarity_matrix[-1:, :].repeat(
                        predicted_interaction.shape[0] - similarity_matrix.shape[0], 1)
                    similarity_matrix = torch.cat([similarity_matrix, pad_row], dim=0)
                    pad_col = similarity_matrix[:, -1:].repeat(1,
                                                               predicted_interaction.shape[1] - similarity_matrix.shape[
                                                                   1])
                    similarity_matrix = torch.cat([similarity_matrix, pad_col], dim=1)

            # 使用MSE损失计算一致性
            # 确保两个张量形状完全一致后再计算损失
            if predicted_interaction.shape != similarity_matrix.shape:
                # 进行最终的形状检查和调整
                min_rows = min(predicted_interaction.shape[0], similarity_matrix.shape[0])
                min_cols = min(predicted_interaction.shape[1], similarity_matrix.shape[1])
                predicted_interaction = predicted_interaction[:min_rows, :min_cols]
                similarity_matrix = similarity_matrix[:min_rows, :min_cols]

            physics_loss = F.mse_loss(predicted_interaction, similarity_matrix)
            total_physics_loss += weights.get(physics_type, 1.0) * physics_loss

        return total_physics_loss


def test_physics_loss_direct():
    """
    测试直接物理损失模块
    """
    # 创建测试数据
    batch_size = 32
    num_features_lncrna = 1  # 只使用疏水性特征
    num_features_protein = 1  # 只使用疏水性特征

    lncrna_physics = torch.randn(batch_size, num_features_lncrna)
    protein_physics = torch.randn(batch_size, num_features_protein)
    predicted_interaction = torch.sigmoid(torch.randn(batch_size, batch_size))

    # 测试直接物理损失模块
    physics_loss_direct = PhysicsLossDirect()
    physical_matrices = physics_loss_direct.compute_physical_matrices(lncrna_physics, protein_physics)
    total_loss = physics_loss_direct.compute_physics_loss(predicted_interaction, physical_matrices)

    print("直接物理损失模块测试:")
    print(f"lncRNA物理特征形状: {lncrna_physics.shape}")
    print(f"蛋白质物理特征形状: {protein_physics.shape}")
    print(f"预测交互矩阵形状: {predicted_interaction.shape}")
    print(f"物理矩阵数量: {len(physical_matrices)}")
    for name, matrix in physical_matrices.items():
        print(f"  {name}矩阵形状: {matrix.shape}")
    print(f"总的物理一致性损失: {total_loss.item()}")


if __name__ == "__main__":
    test_physics_loss_direct()