import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    交叉注意力机制的另一种实现方式
    使用可学习的查询、键、值机制
    """

    def __init__(self, feature_dim=128, head_num=8):
        """
        初始化交叉注意力模块

        Args:
            feature_dim (int): 输入特征的维度
            head_num (int): 注意力头的数量
        """
        super(CrossAttention, self).__init__()
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.head_dim = feature_dim // head_num

        # 确保特征维度可以被头数整除
        assert self.head_dim * head_num == feature_dim, "feature_dim必须能被head_num整除"

        # 多头注意力的线性变换层
        self.lncrna_q_proj = nn.Linear(feature_dim, feature_dim)
        self.protein_k_proj = nn.Linear(feature_dim, feature_dim)
        self.protein_v_proj = nn.Linear(feature_dim, feature_dim)

        self.protein_q_proj = nn.Linear(feature_dim, feature_dim)
        self.lncrna_k_proj = nn.Linear(feature_dim, feature_dim)
        self.lncrna_v_proj = nn.Linear(feature_dim, feature_dim)

        self.output_projection_lncrna = nn.Linear(feature_dim, feature_dim)
        self.output_projection_protein = nn.Linear(feature_dim, feature_dim)

    def forward(self, lncrna_features, protein_features):
        """
        前向传播，计算交叉注意力

        Args:
            lncrna_features (torch.Tensor): lncRNA特征，形状为[num_lncrna, feature_dim]
            protein_features (torch.Tensor): 蛋白质特征，形状为[num_protein, feature_dim]

        Returns:
            tuple: (weighted_lncrna, weighted_protein)
                weighted_lncrna (torch.Tensor): 加权后的lncRNA特征，形状为[num_lncrna, feature_dim]
                weighted_protein (torch.Tensor): 加权后的蛋白质特征，形状为[num_protein, feature_dim]
        """
        num_lncrna = lncrna_features.size(0)
        num_protein = protein_features.size(0)

        # 多头注意力计算
        # lncRNA关注蛋白质
        # 使用detach().clone()避免计算图重复使用的问题
        lncrna_q = self.lncrna_q_proj(lncrna_features).view(num_lncrna, self.head_num, self.head_dim).transpose(0, 1)
        protein_k = self.protein_k_proj(protein_features).view(num_protein, self.head_num, self.head_dim).transpose(0, 1)
        protein_v = self.protein_v_proj(protein_features).view(num_protein, self.head_num, self.head_dim).transpose(0, 1)

        # 计算注意力分数
        # [head_num, num_lncrna, head_dim] * [head_num, head_dim, num_protein] -> [head_num, num_lncrna, num_protein]
        attention_scores_l2p = torch.matmul(lncrna_q, protein_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights_l2p = F.softmax(attention_scores_l2p, dim=-1)

        # 应用注意力权重
        # [head_num, num_lncrna, num_protein] * [head_num, num_protein, head_dim] -> [head_num, num_lncrna, head_dim]
        weighted_lncrna = torch.matmul(attention_weights_l2p, protein_v)
        # [head_num, num_lncrna, head_dim] -> [num_lncrna, head_num, head_dim] -> [num_lncrna, feature_dim]
        weighted_lncrna = weighted_lncrna.transpose(0, 1).contiguous().view(num_lncrna, self.feature_dim)
        weighted_lncrna = self.output_projection_lncrna(weighted_lncrna)

        # 蛋白质关注lncRNA
        protein_q = self.protein_q_proj(protein_features).view(num_protein, self.head_num, self.head_dim).transpose(0, 1)
        lncrna_k = self.lncrna_k_proj(lncrna_features).view(num_lncrna, self.head_num, self.head_dim).transpose(0, 1)
        lncrna_v = self.lncrna_v_proj(lncrna_features).view(num_lncrna, self.head_num, self.head_dim).transpose(0, 1)

        # 计算注意力分数
        # [head_num, num_protein, head_dim] * [head_num, head_dim, num_lncrna] -> [head_num, num_protein, num_lncrna]
        attention_scores_p2l = torch.matmul(protein_q, lncrna_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights_p2l = F.softmax(attention_scores_p2l, dim=-1)

        # 应用注意力权重
        # [head_num, num_protein, num_lncrna] * [head_num, num_lncrna, head_dim] -> [head_num, num_protein, head_dim]
        weighted_protein = torch.matmul(attention_weights_p2l, lncrna_v)
        # [head_num, num_protein, head_dim] -> [num_protein, head_num, head_dim] -> [num_protein, feature_dim]
        weighted_protein = weighted_protein.transpose(0, 1).contiguous().view(num_protein, self.feature_dim)
        weighted_protein = self.output_projection_protein(weighted_protein)

        return weighted_lncrna, weighted_protein


def test_cross_attention():
    """
    测试交叉注意力机制
    """
    # 创建测试数据
    lncrna_features = torch.randn(1073, 128)
    protein_features = torch.randn(73, 128)

    crossAttention = CrossAttention()
    weighted_lncrna_v2, weighted_protein_v2 = crossAttention(lncrna_features, protein_features)

    print("\nV2版本:")
    print("加权后lncRNA特征形状:", weighted_lncrna_v2.shape)
    print("加权后蛋白质特征形状:", weighted_protein_v2.shape)

    # 验证形状是否正确
    assert weighted_lncrna_v2.shape == lncrna_features.shape, "V2版本lncRNA特征形状不正确"
    assert weighted_protein_v2.shape == protein_features.shape, "V2版本蛋白质特征形状不正确"

    print("测试通过!")


if __name__ == "__main__":
    test_cross_attention()