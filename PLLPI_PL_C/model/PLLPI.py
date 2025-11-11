import torch
import torch.nn as nn
import torch.nn.functional as F
from .CrossAttention import CrossAttention


class DeepFeatureExtractor(nn.Module):
    """
    使用1D卷积神经网络提取深度特征，包含8层卷积层和残差连接
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=8, dropout_rate=0.2):
        super(DeepFeatureExtractor, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 第一层卷积
        self.conv_in = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)

        # 8层卷积层，每层之间使用残差连接
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            for _ in range(num_layers)
        ])

        # 残差连接的适配层（如果需要改变维度）
        self.residual_adapt = nn.Conv1d(input_dim, hidden_dim, kernel_size=1) if input_dim != hidden_dim else None

        # 池化层，将序列自适应池化到固定长度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)

        # 最终特征映射层
        self.final_mapping = nn.Linear(hidden_dim * 10, 128)

        # 添加dropout层
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        前向传播

        参数:
        x: 输入特征，形状为(batch_size, feature_dim, sequence_length)

        返回:
        输出特征，形状为(batch_size, 128)
        """
        # 确保输入是3维的
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # 添加序列维度

        # 如果输入通道数与隐藏层通道数不同，需要适配
        if self.residual_adapt is not None:
            residual = self.residual_adapt(x)
        else:
            residual = x if self.input_dim == self.hidden_dim else x

        # 第一层卷积
        out = F.relu(self.conv_in(x))

        # 8层卷积层，每层后添加残差连接
        for i in range(self.num_layers):
            # 保存当前输出用于残差连接
            identity = out

            # 卷积层
            out = self.conv_layers[i](out)
            out = F.relu(out)

            # 添加dropout
            out = self.dropout(out)

            # 添加残差连接
            # 需要确保维度匹配才能相加
            if identity.size(1) == out.size(1):  # 通道数匹配
                out = out + identity
            elif i == 0 and residual.size(1) == out.size(1):  # 第一层后使用初始残差
                out = out + residual

        # 自适应池化
        out = self.adaptive_pool(out)  # (batch_size, hidden_dim, 10)

        # 展平并映射到最终维度
        out = out.view(out.size(0), -1)  # (batch_size, hidden_dim * 10)
        out = self.final_mapping(out)  # (batch_size, 128)

        return out


class PLLPI_Physics(nn.Module):
    """
    lncRNA-蛋白质相互作用预测模型（支持物理特征）
    """

    def __init__(self, lncrna_dim=128, protein_dim=128, lncrna_physics_dim=5, protein_physics_dim=5,
                 hidden_dim=64, dropout_rate=0.2, use_physics=True):
        super(PLLPI_Physics, self).__init__()
        self.lncrna_dim = lncrna_dim
        self.protein_dim = protein_dim
        self.lncrna_physics_dim = lncrna_physics_dim
        self.protein_physics_dim = protein_physics_dim
        self.use_physics = use_physics

        # 深度特征提取器
        self.lncrna_feature_extractor = DeepFeatureExtractor(lncrna_dim, hidden_dim, dropout_rate=dropout_rate)
        self.protein_feature_extractor = DeepFeatureExtractor(protein_dim, hidden_dim, dropout_rate=dropout_rate)

        # 交叉注意力机制
        self.cross_attention = CrossAttention(feature_dim=128, head_num=8)

        # 添加一个交互建模层，用于学习lncRNA和蛋白质之间的相互作用
        self.interaction_modeling = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # 交互预测层
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),  # 交互建模后的特征
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, batch_data):
        """
        前向传播

        参数:
        batch_data: 包含lncrna_features、protein_features、lncrna_physics_features和protein_physics_features的批次数据

        返回:
        预测结果
        """
        # 获取特征
        lncrna_features = batch_data['lncrna_features']  # (batch_size, 128)
        protein_features = batch_data['protein_features']  # (batch_size, 128)

        # 添加维度以适应卷积层 (batch_size, feature_dim, sequence_length)
        lncrna_features_conv = lncrna_features.unsqueeze(-1)  # (batch_size, 128, 1)
        protein_features_conv = protein_features.unsqueeze(-1)  # (batch_size, 128, 1)

        # 深度特征提取
        lncrna_deep_features = self.lncrna_feature_extractor(lncrna_features_conv)  # (batch_size, 128)
        protein_deep_features = self.protein_feature_extractor(protein_features_conv)  # (batch_size, 128)

        # 应用交叉注意力机制
        lncrna_weighted, protein_weighted = self.cross_attention(lncrna_deep_features, protein_deep_features)

        # 特征拼接
        combined_features = torch.cat([lncrna_weighted, protein_weighted], dim=1)  # (batch_size, 256)

        # 交互建模
        interaction_features = self.interaction_modeling(combined_features)  # (batch_size, 128)

        # 预测
        output = self.predictor(interaction_features).squeeze()  # (batch_size,)
        # print( 'output:', output)

        return output

    def get_embeddings_and_physics(self, batch_data):
        """
        获取embedding和物理特征，用于物理损失计算

        参数:
        batch_data: 包含lncrna_features、protein_features、lncrna_physics_features和protein_physics_features的批次数据

        返回:
        tuple: (lncrna_embeddings, protein_embeddings, lncrna_physics, protein_physics)
        """
        # 获取特征
        lncrna_features = batch_data['lncrna_features']  # (batch_size, 128)
        protein_features = batch_data['protein_features']  # (batch_size, 128)
        lncrna_physics = batch_data.get('lncrna_physics_features', None)  # (batch_size, physics_dim)
        protein_physics = batch_data.get('protein_physics_features', None)  # (batch_size, physics_dim)

        # 添加维度以适应卷积层 (batch_size, feature_dim, sequence_length)
        lncrna_features_conv = lncrna_features.unsqueeze(-1)  # (batch_size, 128, 1)
        protein_features_conv = protein_features.unsqueeze(-1)  # (batch_size, 128, 1)

        # 深度特征提取
        lncrna_deep_features = self.lncrna_feature_extractor(lncrna_features_conv)  # (batch_size, 128)
        protein_deep_features = self.protein_feature_extractor(protein_features_conv)  # (batch_size, 128)

        # 应用交叉注意力机制
        lncrna_weighted, protein_weighted = self.cross_attention(lncrna_deep_features, protein_deep_features)

        return lncrna_weighted, protein_weighted, lncrna_physics, protein_physics