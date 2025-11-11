import csv
import pandas as pd
import torch
import numpy as np
from model.Generate_Heterogeneous_Graph import Generate_Heterogeneous_Graph
import utils
from torch.utils.data import Dataset, DataLoader
import os
import random


def custom_collate_fn(batch):
    """
    自定义collate函数，用于处理变长序列数据
    """
    # 将批次中的各个样本组合在一起
    batch_dict = {}
    for key in batch[0].keys():
        if key in ['lncrna_idx', 'protein_idx', 'lncrna_features', 'protein_features', 'label']:
            # 这些是固定大小的张量，可以直接堆叠
            batch_dict[key] = torch.stack([sample[key] for sample in batch])
        elif key in ['lncrna_physics_features', 'protein_physics_features']:
            # 物理特征也是固定大小的张量，可以直接堆叠
            batch_dict[key] = torch.stack([sample[key] for sample in batch])

    return batch_dict


class Load_data(object):
    def __init__(self, data_path, batch_size, args, aggregated_feature_heterogeneous_graph_data, train):
        self.data_path = data_path
        self.batch_size = batch_size
        self.args = args
        self.train = train
        self.aggregated_feature_heterogeneous_graph_data = aggregated_feature_heterogeneous_graph_data
        data = self.load_data(data_path)
        # 获取异构图数据

        # 构建所有的负样本
        '''
            如果修改为 self.generate_negative_sampling(data, self.positive_pairs)，
            则传入的是：data=self.load_data(data_path) 返回的原始数据,即通过 pd.read_csv() 读取的 DataFrame 数据
        '''
        # 时间有点长，直接使用预先采样并保存好的负样本，如果是第一次的话需要执行负采样步骤
        negative_samples_file = r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI\output\save_gengrate_negative_samples\negative_samples.csv'

        # 获取从lncrna到protein的边索引,以便获取行列索引
        edge_index_lncrna_to_protein = self.aggregated_feature_heterogeneous_graph_data[
            'lncrna', 'interaction', 'protein'].edge_index
        # 获取行列索引，以便获取所有的正样本对
        lncrna_indices = edge_index_lncrna_to_protein[0]  # lncrna节点索引
        protein_indices = edge_index_lncrna_to_protein[1]  # protein节点索引
        self.positive_pairs = []
        self.positive_pairs.append((lncrna_indices, protein_indices))

        # 检查负样本文件是否存在，如果存在则直接加载，否则执行采样步骤
        if os.path.exists(negative_samples_file):
            print("检测到预先生成的负样本文件，直接加载...")
            self.negative_pairs = self.load_neagtive_samples_data()
        else:
            print("未检测到负样本文件，开始执行负采样...")
            self.negative_pairs = self.generate_negative_sampling(
                distance_threshold=3
            )

    def load_data(self, data_path):
        interaction_lable_data = pd.read_csv(
            r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI\dataset\data1\lncRNA_protein_interaction_matrix.csv',
            index_col=0, header=0)
        # print('interaction_lable_data:', interaction_lable_data)
        # print('interaction_lable_data.shape:', interaction_lable_data.shape)
        return interaction_lable_data

    def generate_negative_sampling(self, distance_threshold):
        # 用于保存所有的负样本对
        negative_pairs = []

        total_lncrna_nodes = self.aggregated_feature_heterogeneous_graph_data['lncrna'].num_nodes
        total_protein_nodes = self.aggregated_feature_heterogeneous_graph_data['protein'].num_nodes

        # 获取从lncrna到protein的边索引,以便获取行列索引
        edge_index_lncrna_to_protein = self.aggregated_feature_heterogeneous_graph_data[
            'lncrna', 'interaction', 'protein'].edge_index
        # 获取行列索引，以便获取所有的正样本对
        lncrna_indices = edge_index_lncrna_to_protein[0]  # lncrna节点索引
        protein_indices = edge_index_lncrna_to_protein[1]  # protein节点索引
        positive_pairs = []
        positive_pairs.append((lncrna_indices, protein_indices))

        # 解包positive_pairs获取行索引和列索引数组
        row_indices, col_indices = positive_pairs[0]
        # 创建一个集合来存储正样本对
        positive_set = set(zip(row_indices, col_indices))

        # 添加进度条
        total_iterations = total_lncrna_nodes * total_protein_nodes
        processed = 0
        last_percentage = 0

        print('开始负采样...')
        for lncrna_index in range(total_lncrna_nodes):
            for protein_index in range(total_protein_nodes):
                # 检查是否是正样本对
                if (lncrna_index, protein_index) not in positive_set:
                    # 计算节点之间的距离
                    node_distance = utils.calculate_lncrna_to_protein_node_distance(
                        self.aggregated_feature_heterogeneous_graph_data,
                        lncrna_index,
                        protein_index,
                    )
                    # 如果距离超出阈值，则认为是负样本
                    if node_distance > distance_threshold:
                        negative_pairs.append((lncrna_index, protein_index))

                # 更新进度
                processed += 1
                current_percentage = (processed * 100) // total_iterations
                # last_percentage:上一次显示的进度百分比值
                if current_percentage > last_percentage:
                    print(f'\r负采样进度: {current_percentage}%', end='', flush=True)
                    last_percentage = current_percentage

        print('\n负采样完成')
        # 保存到CSV文件
        with open(
                r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI\output\save_gengrate_negative_samples\negative_samples.csv',
                'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['lncrna_index', 'protein_index'])  # 表头
            for pair in negative_pairs:
                writer.writerow(pair)

        print('\n负采样完成并保存到文件')

        return negative_pairs

    def load_neagtive_samples_data(self):
        # 读取CSV文件
        negative_samples_data = pd.read_csv(
            r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI\output\save_gengrate_negative_samples\negative_samples.csv',
            header=0)
        # print('negative_samples_data:', negative_samples_data)
        # print('negative_samples_data.shape:', negative_samples_data.shape)
        return negative_samples_data

    def generate_train_data_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        # 执行这个代码需要进行负采样的过程，如果是第一次需要执行，如果已经执行并保存了负采样之后的样本，则直接执行下面的，也就是直接加载保存的负样本
        # dataset=generate_postive_negative_labels(self.positive_pairs,self.negative_pairs)
        # 如果已经执行并保存了负采样之后的样本，则直接执行这个，也就是直接加载保存的负样本
        if isinstance(self.negative_pairs, pd.DataFrame):
            negative_pairs = self.negative_pairs
        else:
            negative_pairs = self.load_neagtive_samples_data()

        # 根据正样本的数量，从所有的负样本中随机选取负样本
        # 获取正样本数量
        pos_row_indices, pos_col_indices = self.positive_pairs[0]
        positive_count = len(pos_row_indices)

        # 负样本平衡处理已移至generate_postive_negative_labels类中进行，避免重复处理

        dataset = generate_postive_negative_labels(self.positive_pairs, negative_pairs, self.args,
                                                   self.aggregated_feature_heterogeneous_graph_data, self.train)
        '''
            当DataLoader处理数据时，它会将多个样本组合成一个批次。例如，如果batch_size是64，那么：
            batch_data['lncrna_idx'] 是一个包含64个lncRNA索引的张量，形状为 [64]
            batch_data['protein_idx'] 是一个包含64个protein索引的张量，形状为 [66]
            batch_data['lncrna_features'] 是所有lncRNA的特征矩阵，形状为 [总lncRNA数量, 特征维度]
            batch_data['protein_features'] 是所有protein的特征矩阵，形状为 [总protein数量, 特征维度]

            索引是batch_size个索引，特征是总特征，然后根据batch_size个索引从总特征中选择batch_size个特征
        '''
        if self.train:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,  # 启用数据打乱
                drop_last=False,
                collate_fn=custom_collate_fn  # 使用自定义的collate函数
            )
            return dataloader
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,  # 不打乱数据
                drop_last=False,
                collate_fn=custom_collate_fn  # 使用自定义的collate函数
            )
            return dataloader


class generate_postive_negative_labels(Dataset):
    def __init__(self, positive_pairs, negative_pairs, args, aggregated_feature_heterogeneous_graph_data,
                 is_train=True):
        # 解包positive_pairs获取行索引和列索引数组
        pos_row_indices, pos_col_indices = positive_pairs[0]
        # 根据负样本的数据类型进行相应处理
        # 转换为元组列表
        negative_pairs_list = [tuple(x) for x in negative_pairs.values.tolist()]

        # 在划分数据集之前进行负样本平衡处理
        positive_count = len(pos_row_indices)
        if len(negative_pairs_list) > positive_count:
            # 随机选择与正样本数量相同的负样本
            sampled_indices = random.sample(range(len(negative_pairs_list)), positive_count)
            negative_pairs_list = [negative_pairs_list[i] for i in sampled_indices]

        # 组合正负样本
        balanced_pairs = list(zip(pos_row_indices, pos_col_indices)) + negative_pairs_list
        balanced_labels = [1] * len(pos_row_indices) + [0] * len(negative_pairs_list)

        # 划分训练集和测试集 (80% 训练, 20% 测试)
        total_samples = len(balanced_pairs)
        indices = list(range(total_samples))

        # 固定随机种子以确保训练集和测试集的划分一致
        random.seed(42)
        random.shuffle(indices)

        split_idx = int(0.8 * total_samples)
        if is_train:
            # 训练集使用前80%的数据
            self.pairs = [balanced_pairs[i] for i in indices[:split_idx]]
            self.labels = [balanced_labels[i] for i in indices[:split_idx]]
        else:
            # 测试集使用后20%的数据
            self.pairs = [balanced_pairs[i] for i in indices[split_idx:]]
            self.labels = [balanced_labels[i] for i in indices[split_idx:]]

        # 打印数据集信息用于调试
        positive_count = sum(self.labels)  # sum()是求和，不是求长度  求label中1的数量之和，就是正样本的数量
        negative_count = len(self.labels) - positive_count
        dataset_type = '训练' if is_train else '测试'
        print(f"{dataset_type}集 - 总样本数: {len(self.labels)}, 正样本: {positive_count}, 负样本: {negative_count}")

        # 添加额外的调试信息
        if len(self.labels) > 0:
            print(f"{dataset_type}集 - 标签分布: {np.unique(self.labels, return_counts=True)}")

        self.args = args
        # 保存聚合特征并进行标准化
        lncrna_features = aggregated_feature_heterogeneous_graph_data['lncrna'].x
        protein_features = aggregated_feature_heterogeneous_graph_data['protein'].x

        # 对特征进行Z-score标准化
        # 计算训练集的均值和标准差（防止除零错误添加小常数1e-8）
        lncrna_mean = lncrna_features.mean(dim=0)
        lncrna_std = lncrna_features.std(dim=0) + 1e-8
        protein_mean = protein_features.mean(dim=0)
        protein_std = protein_features.std(dim=0) + 1e-8

        # 标准化所有数据（训练集和测试集）
        self.lncrna_features = (lncrna_features - lncrna_mean) / lncrna_std
        self.protein_features = (protein_features - protein_mean) / protein_std

        print("特征标准化完成")
        print(f"lncRNA特征维度: {self.lncrna_features.shape}")
        print(f"protein特征维度: {self.protein_features.shape}")

        # 初始化物理特征（如果可用）
        self.lncrna_physics_features = None
        self.protein_physics_features = None

        # 尝试从指定路径加载物理特征
        if hasattr(args, 'lncrna_physics_path') and hasattr(args, 'protein_physics_path'):
            if os.path.exists(args.lncrna_physics_path) and os.path.exists(args.protein_physics_path):
                print(f"从指定路径加载lncRNA物理特征: {args.lncrna_physics_path}")
                lncrna_physics_data = pd.read_csv(args.lncrna_physics_path, index_col=0, header=0)
                # print('lncrna_physics_data:',lncrna_physics_data.head())
                print(f"从指定路径加载蛋白质物理特征: {args.protein_physics_path}")
                protein_physics_data = pd.read_csv(args.protein_physics_path, index_col=0, header=0)
                # print('protein_physics_data:',protein_physics_data.head())

                # 根据图中使用的蛋白质索引选择对应的物理特征
                if hasattr(aggregated_feature_heterogeneous_graph_data, 'protein_index_map'):
                    # 获取图中实际使用的蛋白质索引（原始索引）
                    used_protein_indices = sorted(aggregated_feature_heterogeneous_graph_data.protein_index_map.keys())
                    # 从物理特征数据中选择这些蛋白质对应的特征
                    protein_physics_data = protein_physics_data.iloc[used_protein_indices]
                    print(f"根据图结构筛选后，仅使用 {len(used_protein_indices)} 个蛋白质的物理特征")

                # 转换为张量前确保数据类型正确
                # 先选择数值列，然后转换为float32
                lncrna_physics_values = lncrna_physics_data.select_dtypes(include=[np.number]).astype(np.float32).values
                protein_physics_values = protein_physics_data.select_dtypes(include=[np.number]).astype(
                    np.float32).values

                # 转换为张量
                self.lncrna_physics_features = torch.tensor(lncrna_physics_values, dtype=torch.float32)
                self.protein_physics_features = torch.tensor(protein_physics_values, dtype=torch.float32)

                print(f"lncRNA物理特征维度: {self.lncrna_physics_features.shape}")
                print(f"蛋白质物理特征维度: {self.protein_physics_features.shape}")
                # 打印部分物理特征值用于调试
                # print(f"lncRNA物理特征样例: {self.lncrna_physics_features[:3]}")
                # print(f"蛋白质物理特征样例: {self.protein_physics_features[:3]}")
            else:
                print("指定的物理特征文件路径不存在，使用默认特征提取方法")
                self._extract_physics_features_from_main_features()
        else:
            print("未指定物理特征文件路径，使用默认特征提取方法")
            self._extract_physics_features_from_main_features()

    def _extract_physics_features_from_main_features(self):
        """
        从主特征中提取物理特征的默认方法
        """
        # 使用默认路径加载物理特征
        default_lncrna_physics_path = r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI_PL_B\physics_loss\output\original_physics_feature\original_lncrna_feature.csv'
        default_protein_physics_path = r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI_PL_B\physics_loss\output\original_physics_feature\original_protein_feature.csv'

        try:
            if os.path.exists(default_lncrna_physics_path) and os.path.exists(default_protein_physics_path):
                print("从默认路径加载物理特征...")
                # 加载lncRNA物理特征
                lncrna_physics_data = pd.read_csv(default_lncrna_physics_path, index_col=0)
                # 加载蛋白质物理特征
                protein_physics_data = pd.read_csv(default_protein_physics_path, index_col=0)

                # 选择数值列并转换为float32
                lncrna_physics_values = lncrna_physics_data.select_dtypes(include=[np.number]).astype(np.float32).values
                protein_physics_values = protein_physics_data.select_dtypes(include=[np.number]).astype(
                    np.float32).values

                # 转换为张量
                self.lncrna_physics_features = torch.tensor(lncrna_physics_values, dtype=torch.float32)
                self.protein_physics_features = torch.tensor(protein_physics_values, dtype=torch.float32)

                print(f"lncRNA物理特征维度: {self.lncrna_physics_features.shape}")
                print(f"蛋白质物理特征维度: {self.protein_physics_features.shape}")
            else:
                print("默认物理特征文件不存在，创建零特征张量")
                # 创建零特征张量
                self.lncrna_physics_features = torch.zeros(self.lncrna_features.size(0), 3, dtype=torch.float32)
                self.protein_physics_features = torch.zeros(self.protein_features.size(0), 4, dtype=torch.float32)
        except Exception as e:
            print(f"加载物理特征时出错: {e}")
            # 出错时创建零特征张量
            self.lncrna_physics_features = torch.zeros(self.lncrna_features.size(0), 3, dtype=torch.float32)
            self.protein_physics_features = torch.zeros(self.protein_features.size(0), 4, dtype=torch.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lncrna_idx, protein_idx = self.pairs[idx]  # 4420
        # print('len(self.pairs):',len(self.pairs))
        label = self.labels[idx]
        # print('label:',label)

        # 检查索引范围，防止越界
        max_lncrna_idx = self.lncrna_features.size(0) - 1
        max_protein_idx = self.protein_features.size(0) - 1

        # 确保索引在有效范围内
        lncrna_idx = max(0, min(lncrna_idx, max_lncrna_idx))
        protein_idx = max(0, min(protein_idx, max_protein_idx))

        # 安全地创建张量并移动到设备
        # 使用推荐的方法创建张量
        lncrna_idx_tensor = torch.as_tensor(lncrna_idx, dtype=torch.long).to(self.args.device)
        protein_idx_tensor = torch.as_tensor(protein_idx, dtype=torch.long).to(self.args.device)
        label_tensor = torch.as_tensor(label, dtype=torch.long).to(self.args.device)

        # 获取特征张量并移动到设备
        # 根据索引获取对应的特征向量，而不是传递整个特征矩阵
        # 使用detach().clone()避免计算图重复使用的问题
        lncrna_features_tensor = self.lncrna_features[lncrna_idx].detach().clone().to(self.args.device)
        protein_features_tensor = self.protein_features[protein_idx].detach().clone().to(self.args.device)

        sample = {
            # 在PyTorch中，张量默认创建在CPU上，除非显式指定设备。
            'lncrna_idx': lncrna_idx_tensor,
            'protein_idx': protein_idx_tensor,
            'lncrna_features': lncrna_features_tensor,
            'protein_features': protein_features_tensor,
            'label': label_tensor
        }

        # 如果有物理特征，则添加到样本中
        if self.lncrna_physics_features is not None:
            lncrna_physics_tensor = self.lncrna_physics_features[lncrna_idx].detach().clone().to(self.args.device)
            sample['lncrna_physics_features'] = lncrna_physics_tensor
            # 打印lncrna物理特征信息用于调试
            # print(f"lncrna物理特征已添加到样本中，数据: {lncrna_physics_tensor}")

        if self.protein_physics_features is not None:
            protein_physics_tensor = self.protein_physics_features[protein_idx].detach().clone().to(self.args.device)
            sample['protein_physics_features'] = protein_physics_tensor
            # 打印protein物理特征信息用于调试
            # print(f"protein物理特征已添加到样本中，数据: {protein_physics_tensor}")

        return sample
