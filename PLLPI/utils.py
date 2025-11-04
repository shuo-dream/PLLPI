import networkx as nx
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
import numpy as np
import torch


# 计算一对节点之间的最短距离
def calculate_lncrna_to_protein_node_distance(heterogeneous_graph_data, lncRNA_idx, protein_idx):
    # 将异构图转换为NetworkX图以便计算最短路径
    G = nx.Graph()

    # 添加节点
    for i in range(heterogeneous_graph_data['lncrna'].num_nodes):
        G.add_node(f'lncrna_{i}', type='lncrna')
    for i in range(heterogeneous_graph_data['protein'].num_nodes):
        G.add_node(f'protein_{i}', type='protein')

    # 添加边
    edge_index_lncrna_to_protein = heterogeneous_graph_data['lncrna', 'interaction', 'protein'].edge_index
    for i in range(edge_index_lncrna_to_protein.shape[1]):
        lncrna_node = f'lncrna_{edge_index_lncrna_to_protein[0][i].item()}'
        protein_node = f'protein_{edge_index_lncrna_to_protein[1][i].item()}'
        # 检查索引是否超出范围
        if edge_index_lncrna_to_protein[0][i].item() >= heterogeneous_graph_data['lncrna'].num_nodes or \
                edge_index_lncrna_to_protein[1][i].item() >= heterogeneous_graph_data['protein'].num_nodes:
            print(
                f"警告：索引超出范围，lncrna索引: {edge_index_lncrna_to_protein[0][i].item()}, protein索引: {edge_index_lncrna_to_protein[1][i].item()}")
            continue
        G.add_edge(lncrna_node, protein_node)

    # 计算并保存(因为外面已经写好了循环，所以不用保存)每个rna到protein的最短路径,调用此函数的函数已经写好了循环
    source = f'lncrna_{lncRNA_idx}'
    target = f'protein_{protein_idx}'
    try:
        distance = nx.shortest_path_length(G, source, target)
        return float(distance)
    except nx.NetworkXNoPath:
        return float('inf')

# 设置随机种子
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 添加调试函数，用于检查图结构
def debug_graph_structure(heterogeneous_graph_data):
    """
    调试图结构，打印图的基本信息
    """
    print("=== 图结构调试信息 ===")
    print(f"lncrna节点数量: {heterogeneous_graph_data['lncrna'].num_nodes}")
    print(f"protein节点数量: {heterogeneous_graph_data['protein'].num_nodes}")

    if ('lncrna', 'interaction', 'protein') in heterogeneous_graph_data.edge_types:
        edge_index = heterogeneous_graph_data['lncrna', 'interaction', 'protein'].edge_index
        print(f"lncrna->protein边数量: {edge_index.shape[1]}")
        # 安全地检查索引范围
        try:
            edge_index_cpu = edge_index.cpu()
            print(f"lncrna索引范围: [{edge_index_cpu[0].min().item()}, {edge_index_cpu[0].max().item()}]")
            print(f"protein索引范围: [{edge_index_cpu[1].min().item()}, {edge_index_cpu[1].max().item()}]")
        except Exception as e:
            print(f"无法检查索引范围: {e}")

    if ('protein', 'interaction', 'lncrna') in heterogeneous_graph_data.edge_types:
        edge_index = heterogeneous_graph_data['protein', 'interaction', 'lncrna'].edge_index
        print(f"protein->lncrna边数量: {edge_index.shape[1]}")
        # 安全地检查索引范围
        try:
            edge_index_cpu = edge_index.cpu()
            print(f"protein索引范围: [{edge_index_cpu[0].min().item()}, {edge_index_cpu[0].max().item()}]")
            print(f"lncrna索引范围: [{edge_index_cpu[1].min().item()}, {edge_index_cpu[1].max().item()}]")
        except Exception as e:
            print(f"无法检查索引范围: {e}")

    print("====================")


def safe_to_cpu(tensor):
    """
    安全地将张量移动到CPU设备
    """
    try:
        return tensor.cpu()
    except Exception as e:
        print(f"无法将张量移动到CPU: {e}")
        return tensor


def map_protein_index_to_original(heterogeneous_graph_data, mapped_index):
    """
    将映射后的protein索引转换为原始索引
    """
    if hasattr(heterogeneous_graph_data, 'protein_index_map_inverse'):
        return heterogeneous_graph_data.protein_index_map_inverse.get(mapped_index, mapped_index)
    return mapped_index


def map_protein_index_to_mapped(heterogeneous_graph_data, original_index):
    """
    将原始protein索引转换为映射后索引
    """
    if hasattr(heterogeneous_graph_data, 'protein_index_map'):
        return heterogeneous_graph_data.protein_index_map.get(original_index, original_index)
    return original_index


def metrics(y_true, y_pred):
    """
    计算评估指标
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :return: 包含各种评估指标的字典
    """

    # 确保输入是numpy数组
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        # 如果张量需要梯度计算，则先分离梯度计算再转换为numpy数组
        if y_pred.requires_grad:
            y_pred = y_pred.detach().cpu().numpy()
        else:
            y_pred = y_pred.cpu().numpy()

    # 打印标签信息用于调试
    unique_labels, counts = np.unique(y_true, return_counts=True)
    print(f"unique_labels: {unique_labels}, counts: {counts}")

    # 对于概率值，需要转换为二进制预测
    '''
        y_pred > 0.5：这部分会创建一个布尔数组，其中概率大于0.5的元素为True，小于等于0.5的为False
        .astype(int)：将布尔值转换为整数，True变为1，False变为0
    '''
    y_pred_binary = (y_pred > 0.5).astype(int)

    # 打印预测分布用于调试
    unique_pred, pred_counts = np.unique(y_pred_binary, return_counts=True)

    # 计算各种指标
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    # 对于AUC和AUPR，使用原始概率值
    # 检查y_true中是否包含两个类别
    unique_labels = np.unique(y_true)

    # 添加对AUC计算的保护
    if len(unique_labels) < 2:
        auc = 0.5  # 如果只有一类，AUC设为0.5
        aupr = 0.5  # 如果只有一类，AUPR设为0.5
    else:
        auc = roc_auc_score(y_true, y_pred)
        aupr = average_precision_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'aupr': aupr
    }