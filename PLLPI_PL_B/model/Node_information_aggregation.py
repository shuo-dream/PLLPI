import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.nn.conv import GCNConv
from tqdm import tqdm
import os
import numpy as np
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class Node_feature_information_aggregation(nn.Module):
    def __init__(self, args, hidden_dim=128, num_layers=2, agg_type='gat'):
        super(Node_feature_information_aggregation, self).__init__()
        # sage是一种聚合算法,agg_type: 聚合类型 ('sage', 'gat', 'gcn')
        self.device = args.device
        # print('self.device use device:', self.device)
        self.args = args
        self.hidden_dim = hidden_dim
        self.numlayers = num_layers
        self.agg_type = agg_type

        # 存储各层的聚合器
        self.layers = nn.ModuleList()
        self.to(self.device)

    def init_aggregation_layers(self, lncrna_input_dim, protein_input_dim):
        # 清空现有层
        self.layers = nn.ModuleList()

        '''
            第一层和后续层的输入输出维度不完全一样。
            第一层负责将不同维度的输入特征映射到统一的隐藏层维度，而后续层则在统一的隐藏层维度之间进行信息传递。
        '''
        # 第一层，处理不同维度的输入特征
        layer_dict = nn.ModuleDict({
            'lncrna_to_protein': GATConv((lncrna_input_dim, protein_input_dim), self.hidden_dim),
            'protein_to_lncrna': GATConv((protein_input_dim, lncrna_input_dim), self.hidden_dim)
        })
        # 确保模型层在正确的设备上
        layer_dict = layer_dict.to(self.device)
        self.layers.append(layer_dict)

        # 后续层 (后续层输入输出维度相同)
        for _ in range(self.numlayers - 1):
            layer_dict = nn.ModuleDict({
                'lncrna_to_protein': GATConv((self.hidden_dim, self.hidden_dim), self.hidden_dim),
                'protein_to_lncrna': GATConv((self.hidden_dim, self.hidden_dim), self.hidden_dim)
            })
            # 确保模型层在正确的设备上
            layer_dict = layer_dict.to(self.device)
            self.layers.append(layer_dict)

    def forward(self, data):
        data = data.to(self.device)

        # 获取节点特征
        lncrna_features = data['lncrna'].x
        protein_features = data['protein'].x
        # 获取边索引
        edge_index_dict = data.edge_index_dict

        # 多层聚合
        # 基于实际边的计算
        for layer_idx, layer in enumerate(self.layers):
            # 保存当前层的特征用于下一层
            current_lncrna_features = lncrna_features
            current_protein_features = protein_features

            # lncrna到protein的聚合
            if ('lncrna', 'interaction', 'protein') in edge_index_dict:
                edge_index = edge_index_dict[('lncrna', 'interaction', 'protein')]
                # print(f"lncrna nodes count: {data['lncrna'].num_nodes}")
                # print(f"protein nodes count: {data['protein'].num_nodes}")
                # print(f"Edge index shape: {edge_index.shape}")

                # 使用GAT进行特征聚合，不使用自定义注意力权重
                try:
                    aggregated_protein_features = layer['lncrna_to_protein'](
                        (current_lncrna_features, current_protein_features),
                        edge_index
                    )
                    protein_features = aggregated_protein_features
                except Exception as e:
                    print(f"lncrna到protein特征聚合出错: {e}")
                    protein_features = current_protein_features

            # protein到lncrna的聚合
            if ('protein', 'interaction', 'lncrna') in edge_index_dict:
                edge_index = edge_index_dict[('protein', 'interaction', 'lncrna')]
                # print(f"protein nodes count: {data['protein'].num_nodes}")
                # print(f"lncrna nodes count: {data['lncrna'].num_nodes}")
                # print(f"Edge index shape: {edge_index.shape}")

                # 使用GAT进行特征聚合，不使用自定义注意力权重
                try:
                    aggregated_lncrna_features = layer['protein_to_lncrna'](
                        (current_protein_features, current_lncrna_features),
                        edge_index
                    )
                    lncrna_features = aggregated_lncrna_features
                except Exception as e:
                    print(f"protein到lncrna特征聚合出错: {e}")
                    lncrna_features = current_lncrna_features

        # print("lncrna_features", lncrna_features)
        # print("protein_features", protein_features)
        # 更新data对象中的特征
        data['lncrna'].x = lncrna_features
        data['protein'].x = protein_features

        return data


def load_and_aggregate_features(args, heterogeneous_graph_data, lncrna_feature_path=None, protein_feature_path=None,
                                hidden_dim=128, num_layers=2, agg_type='sage'):
    """
    加载特征并执行聚合的便捷函数

    Args:
        args: 参数对象
        heterogeneous_graph_data: HeteroData对象
        lncrna_feature_path: lncRNA特征文件路径
        protein_feature_path: protein特征文件路径
        hidden_dim: 隐藏层维度
        num_layers: 聚合层数
        agg_type: 聚合类型

    Returns:
        聚合后的HeteroData对象
    """
    # 创建聚合器
    aggregator = Node_feature_information_aggregation(
        args,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        agg_type=agg_type
    )

    # 初始化聚合层
    # lncrna和蛋白质初始输入特征的维数
    lncrna_input_dim = heterogeneous_graph_data['lncrna'].x.shape[1]
    # print('lncrna_input_dim:',lncrna_input_dim)
    protein_input_dim = heterogeneous_graph_data['protein'].x.shape[1]
    # print('protein_input_dim:',protein_input_dim)
    aggregator.init_aggregation_layers(lncrna_input_dim, protein_input_dim)

    # 调用 forward 方法
    print("开始特征聚合...")
    '''
        heterogeneous_graph_data 是一个 HeteroData 对象，包含了以下信息：
        节点信息
            lncrna节点：
                heterogeneous_graph_data['lncrna'].num_nodes：lncRNA节点数量
                heterogeneous_graph_data['lncrna'].x：lncRNA节点特征矩阵
            protein节点：
                heterogeneous_graph_data['protein'].num_nodes：蛋白质节点数量
                heterogeneous_graph_data['protein'].x：蛋白质节点特征矩阵
        边信息
            lncrna→protein边：
                heterogeneous_graph_data['lncrna', 'interaction', 'protein'].edge_index：从lncRNA到蛋白质的边索引
            protein→lncrna边：
                heterogeneous_graph_data['protein', 'interaction', 'lncrna'].edge_index：从蛋白质到lncRNA的边索引
        其他信息
            heterogeneous_graph_data.edge_types：图中所有边类型的集合
            heterogeneous_graph_data.protein_index_map：蛋白质原始索引到连续索引的映射
            heterogeneous_graph_data.protein_index_map_inverse：蛋白质连续索引到原始索引的反向映射
    '''
    aggregated_data = aggregator.forward(heterogeneous_graph_data)
    print("特征聚合完成")

    # 安全地打印结果，避免CUDA错误
    # try:
    #     print("聚合后的rna节点特征维度：", aggregated_data['lncrna'].x.shape)
    #     print("聚合后的protein节点特征维度：", aggregated_data['protein'].x.shape)
    # except Exception as e:
    #     print(f"打印聚合特征信息时出错: {e}")

    return aggregated_data
