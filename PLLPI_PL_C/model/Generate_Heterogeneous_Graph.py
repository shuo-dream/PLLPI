import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


class Generate_Heterogeneous_Graph(object):
    def __init__(self, data, args):
        # 因为self.generate_heterogeneous_graph(data)使用到了args,所以args的初始化必须在其前面
        self.args = args
        # 这个data是相互作用标签数据   G:\shen_cong\my\my_project\PLLPI\dataset\data1\lncRNA_protein_interaction_matrix.csv
        self.data = self.generate_heterogeneous_graph(data)

    def generate_heterogeneous_graph(self, data):
        # print('data:',data)
        # print(f"行数: {data.shape[0]}")
        # print(f"列数: {data.shape[1]}")

        lncrna_feature_path = r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI\dataset\data1\original_lncrna_feature.csv'
        protein_feature_path = r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI\dataset\data1\original_protein_feature.csv'
        # 创建异构图对象
        heterogeneous_graph_data = HeteroData()

        # 获取节点名称
        lncran_names = data.index.tolist()
        protein_names = data.columns.tolist()

        # 设置节点数量
        # 修复：根据实际使用的节点数量设置节点数
        # 添加备灾处理
        lncrna_node_count = len(lncran_names)
        unique_protein_indices = set()
        interaction_matrix = data.values
        for i in range(interaction_matrix.shape[0]):
            for j in range(interaction_matrix.shape[1]):
                if interaction_matrix[i, j] == 1:
                    unique_protein_indices.add(j)

        protein_node_count = len(unique_protein_indices)
        # print(f"实际使用的lncrna节点数量: {lncrna_node_count}")
        # # 蛋白质节点的数量是和rna有相互作用的数量  也就是排除掉孤点
        # print(f"实际使用的protein节点数量: {protein_node_count}")

        heterogeneous_graph_data['lncrna'].num_nodes = lncrna_node_count
        heterogeneous_graph_data['protein'].num_nodes = protein_node_count

        # 构建边索引
        # 找到所有的值等于1的rna和蛋白质索引，也就是正样本对
        row_indices, col_indices = np.where(interaction_matrix == 1)

        # 修复：确保protein索引是连续的从0开始
        # 检查protein索引是否连续从0开始
        unique_col_indices = np.unique(col_indices)
        # print(f"原始protein索引范围: [{unique_col_indices.min()}, {unique_col_indices.max()}]")
        # print(f"唯一protein索引数量: {len(unique_col_indices)}")

        # 创建映射以确保索引连续
        protein_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(unique_col_indices))}
        # 保存反向映射，用于后续处理
        self.protein_index_map = protein_index_map
        mapped_col_indices = np.array([protein_index_map[idx] for idx in col_indices])

        # print(f"映射后protein索引范围: [{mapped_col_indices.min()}, {mapped_col_indices.max()}]")

        # 更新row_indices和col_indices
        row_indices = row_indices  # lncrna索引保持不变
        col_indices = mapped_col_indices  # protein索引使用映射后的

        # 加载节点特征，为图数据设置节点特征
        if lncrna_feature_path and protein_feature_path:
            # 加载lncrna特征
            lncrna_features = pd.read_csv(lncrna_feature_path)
            lncrna_features = lncrna_features.values
            lncrna_features_tensor = torch.tensor(lncrna_features, dtype=torch.float32).to(self.args.device)

            # 加载protein特征
            protein_features = pd.read_csv(protein_feature_path)
            protein_features = protein_features.values
            # print('protein_features.shape:',protein_features.shape)
            # 修复：只加载实际使用的protein特征
            if len(unique_col_indices) < len(protein_features):
                protein_features = protein_features[sorted(unique_col_indices)]

            try:
                protein_features_tensor = torch.tensor(protein_features, dtype=torch.float32)
                # 安全地将张量移动到指定设备
                if torch.cuda.is_available() and 'cuda' in str(self.args.device):
                    try:
                        protein_features_tensor = protein_features_tensor.to(self.args.device)
                    except RuntimeError as e:
                        if "device-side assert" in str(e):
                            print("CUDA设备断言错误，尝试在CPU上处理protein特征")
                            protein_features_tensor = protein_features_tensor.cpu()
                        else:
                            raise e
                else:
                    protein_features_tensor = protein_features_tensor.to(self.args.device)
            except Exception as e:
                print(f"创建protein特征张量时出错: {e}")
                raise e

            # 设置节点特征
            heterogeneous_graph_data['lncrna'].x = lncrna_features_tensor
            heterogeneous_graph_data['protein'].x = protein_features_tensor

        # 为图数据设置边
        # 创建双向边(用于计算节点之间的距离)，是有向边
        # 创建张量时默认使用cpu  需要.to(device)  转移到对应的设备上
        edge_index_lncrna_to_protein = torch.tensor(np.array([row_indices, col_indices]), dtype=torch.long).to(
            self.args.device)

        # print('edge_index_lncrna_to_protein use device:', edge_index_lncrna_to_protein.device)
        # print(f"检查lncrna索引范围: [{row_indices.min()}, {row_indices.max()}]")
        # print(f"检查protein索引范围: [{col_indices.min()}, {col_indices.max()}]")

        # 反向边,使用映射后的索引
        edge_index_protein_to_lncrna = torch.tensor(np.array([col_indices, row_indices]), dtype=torch.long).to(
            self.args.device)

        # 将边添加到异构图中
        heterogeneous_graph_data['lncrna', 'interaction', 'protein'].edge_index = edge_index_lncrna_to_protein
        heterogeneous_graph_data['protein', 'interaction', 'lncrna'].edge_index = edge_index_protein_to_lncrna

        heterogeneous_graph_data = heterogeneous_graph_data.to(self.args.device)

        # 添加调试信息
        print("=== 异构图构建完成 ===")
        # print(f"lncrna节点数量: {heterogeneous_graph_data['lncrna'].num_nodes}")
        # print(f"protein节点数量: {heterogeneous_graph_data['protein'].num_nodes}")
        '''
            heterogeneous_graph_data.edge_types 是 HeteroData 类的一个属性，用于存储图中所有边类型的集合。在您的代码中，它包含了以下内容：
            边类型：表示不同节点类型之间的连接关系
            具体值：
            ('lncrna', 'interaction', 'protein')：表示从 lncrna 节点到 protein 节点的边
            ('protein', 'interaction', 'lncrna')：表示从 protein 节点到 lncrna 节点的边
        '''
        if ('lncrna', 'interaction', 'protein') in heterogeneous_graph_data.edge_types:
            '''
                edge_index:
                    形状：(2, num_edges)，其中：
                    第0维：源节点索引
                    第1维：目标节点索
            '''
            # lncrna索引范围: [0, 1096]
            # protein索引范围: [0, 71]
            edge_index = heterogeneous_graph_data['lncrna', 'interaction', 'protein'].edge_index
            # print(f"lncrna->protein边数量: {edge_index.shape[1]}")
            # 安全地检查索引范围
            try:
                edge_index_cpu = edge_index.cpu()
                # print(f"lncrna索引范围: [{edge_index_cpu[0].min().item()}, {edge_index_cpu[0].max().item()}]")
                # print(f"protein索引范围: [{edge_index_cpu[1].min().item()}, {edge_index_cpu[1].max().item()}]")
            except Exception as e:
                print(f"无法检查索引范围: {e}")

        if ('protein', 'interaction', 'lncrna') in heterogeneous_graph_data.edge_types:
            edge_index = heterogeneous_graph_data['protein', 'interaction', 'lncrna'].edge_index
            # print(f"protein->lncrna边数量: {edge_index.shape[1]}")
            # 安全地检查索引范围
            try:
                edge_index_cpu = edge_index.cpu()
                # print(f"protein索引范围: [{edge_index_cpu[0].min().item()}, {edge_index_cpu[0].max().item()}]")
                # print(f"lncrna索引范围: [{edge_index_cpu[1].min().item()}, {edge_index_cpu[1].max().item()}]")
            except Exception as e:
                print(f"无法检查索引范围: {e}")
        print("====================")

        # 保存protein索引映射到图数据中，以便其他模块可以使用
        heterogeneous_graph_data.protein_index_map = protein_index_map
        # 创建反向映射
        # 提供反向映射，将连续索引还原为原始蛋白质索引    原因：便于后续模块查询原始蛋白质ID
        heterogeneous_graph_data.protein_index_map_inverse = {v: k for k, v in protein_index_map.items()}

        # 保存蛋白质名称列表，用于物理特征映射
        heterogeneous_graph_data.protein_names = protein_names
        # 创建从图节点索引到蛋白质名称的映射
        heterogeneous_graph_data.node_idx_to_protein_name = {
            new_idx: protein_names[old_idx]
            for old_idx, new_idx in protein_index_map.items()
        }

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
                heterogeneous_graph_data.protein_names：所有蛋白质名称列表
                heterogeneous_graph_data.node_idx_to_protein_name：图节点索引到蛋白质名称的映射
        '''
        return heterogeneous_graph_data
