# 结合 A 和 B ，既使用融合特征又使用原始特征计算物理损失
import argparse
from operator import index
import pandas as pd
from tqdm import tqdm
import torch
import os
import json
from data_load import Load_data
from model.PLLPI import PLLPI_Physics  # 使用支持物理特征的新模型
from utils import set_random_seeds, debug_graph_structure, metrics
from model.Node_information_aggregation import load_and_aggregate_features
from model.Generate_Heterogeneous_Graph import Generate_Heterogeneous_Graph
# 导入绘图工具
from plot_utils import plot_individual_metrics, plot_loss_curve
# 导入物理损失模块
from physics_loss.physics_loss import PhysicsLossCombined

# 创建参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs_num', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0005)  # 降低学习率以解决极端输出问题
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--dataset', type=str,
                    default=r'G:\shen_cong\my\my_project\my_original_dataset\dataset\datasets\dataset1.txt')
parser.add_argument('--output_dir', type=str, default='output', help='输出目录，用于保存图表等')
parser.add_argument('--warmup_epochs', type=int, default=5, help='warmup预热轮数')
parser.add_argument('--hidden_dim', type=int, default=64, help='隐藏层维度')
parser.add_argument('--num_layers', type=int, default=8, help='卷积层数量')
parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout率')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
parser.add_argument('--load_best_params', type=str, default=None, help='从JSON文件加载最佳超参数')
parser.add_argument('--physics_loss_weight', type=float, default=0.1, help='物理损失权重')
parser.add_argument('--lncrna_physics_path', type=str, default=r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI_PL_C\physics_loss\output\original_physics_feature\original_lncrna_feature.csv', help='lncRNA物理特征文件路径')
parser.add_argument('--protein_physics_path', type=str, default=r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI_PL_C\physics_loss\output\original_physics_feature\original_protein_feature.csv', help='蛋白质物理特征文件路径')

import traceback


def load_best_parameters(args):
    """
    从JSON文件加载最佳超参数

    Args:
        args: 命令行参数对象

    Returns:
        bool: 是否成功加载参数
    """
    if not args.load_best_params:
        return False

    try:
        if not os.path.exists(args.load_best_params):
            print(f"错误: 找不到最佳参数文件 {args.load_best_params}")
            return False

        with open(args.load_best_params, 'r', encoding='utf-8') as f:
            best_params = json.load(f)

        # 更新参数
        args.lr = best_params.get('lr', args.lr)
        args.batch_size = best_params.get('batch_size', args.batch_size)
        args.epochs_num = best_params.get('epochs_num', args.epochs_num)
        args.hidden_dim = best_params.get('hidden_dim', args.hidden_dim)
        args.num_layers = best_params.get('num_layers', args.num_layers)  # 添加缺失的num_layers参数
        args.dropout_rate = best_params.get('dropout_rate', args.dropout_rate)
        args.weight_decay = best_params.get('weight_decay', args.weight_decay)

        print(f"已从 {args.load_best_params} 成功加载最佳超参数:")
        for key, value in best_params.items():
            if key != 'best_f1_score':
                print(f"  {key}: {value}")
        print(f"  对应的最佳F1分数: {best_params.get('best_f1_score', 'N/A')}")

        return True
    except json.JSONDecodeError:
        print(f"错误: 无法解析JSON文件 {args.load_best_params}")
        return False
    except Exception as e:
        print(f"加载最佳参数时出错: {e}")
        return False


# 解析命令行参数
args = parser.parse_args()
# 设置随机种子
set_random_seeds(42)

# 明确区分情况
if args.cuda >= 0 and torch.cuda.is_available():
    device = torch.device(f'cuda:{args.cuda}')
    print(f'使用GPU设备: {device}')
else:
    device = torch.device('cpu')
    print('使用CPU设备')

args.device = device
print(f'使用设备：{device}')


def train(model, optimizer, physics_loss_module, train_batch_data, epoch, warmup_epochs, physics_loss_weight):
    model.train()
    train_loss = 0
    all_labels = []
    all_outputs = []
    # 获取DataLoader的总批次数，用于进度条显示
    total_batches = len(train_batch_data)
    for batch_idx, batch_data in enumerate(train_batch_data):
        # 从批处理数据中提取所需信息
        labels = batch_data['label']

        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(batch_data)
        # 计算主要损失
        main_loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels.float())

        # 计算物理一致性损失
        physics_loss = torch.tensor(0.0, device=device)
        if physics_loss_module is not None:
            # 获取embedding和物理特征
            lncrna_emb, protein_emb, lncrna_physics, protein_physics = model.get_embeddings_and_physics(batch_data)
            # print('lncrna_emb:', lncrna_emb)
            # print('protein_emb:', protein_emb)
            # print('lncrna_physics:', lncrna_physics)
            # print('protein_physics:', protein_physics)
            if lncrna_physics is not None and protein_physics is not None:
                # 计算物理相似度矩阵
                physics_matrices, _, _ = physics_loss_module(lncrna_emb, protein_emb, lncrna_physics, protein_physics)
                # 计算物理损失
                sigmoid_output = torch.sigmoid(output)
                physics_loss = physics_loss_module.compute_physics_loss(sigmoid_output, physics_matrices)

        # 计算总损失
        total_loss = main_loss + physics_loss_weight * physics_loss
        # 计算总损失
        train_loss += total_loss.item()

        # 反向传播
        total_loss.backward()  # 移除retain_graph=True

        # 实现warmup学习率调整
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # 更新参数
        optimizer.step()

        # 收集所有预测和标签用于计算评估指标
        all_labels.append(labels)
        sigmoid_output = torch.sigmoid(output)  # 应用sigmoid获取概率
        all_outputs.append(sigmoid_output)

    # 计算评估指标
    '''
        # 假设有3个批次，每个批次有2个样本
        batch1_labels = torch.tensor([1, 0])  # 第1个批次的标签
        batch2_labels = torch.tensor([1, 1])  # 第2个批次的标签
        batch3_labels = torch.tensor([0, 1])  # 第3个批次的标签

        # 在批次循环中，all_labels列表会变成：
        all_labels = [tensor([1, 0]), tensor([1, 1]), tensor([0, 1])]

        # 执行torch.cat(all_labels, dim=0)后，结果是：
        final_labels = torch.tensor([1, 0, 1, 1, 0, 1])  # 所有标签连接成一个一维张量
    '''
    all_labels = torch.cat(all_labels, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)
    # print('all_outputs:',all_outputs)

    train_metrics = metrics(all_labels, all_outputs)

    return train_loss / len(train_batch_data), train_metrics


def eval(model, optimizer, physics_loss_module, eval_batch_data, physics_loss_weight):
    model.eval()
    total_loss_item = 0
    all_labels = []
    all_outputs = []
    # 获取DataLoader的总批次数，用于进度条显示
    total_batches = len(eval_batch_data)
    # 不进行前向反向传播
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(eval_batch_data):
            labels = batch_data['label']
            output = model(batch_data)
            # 限制输出范围以防止梯度爆炸
            # output = torch.clamp(output, min=-10, max=10)
            main_loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels.float())

            # 计算物理一致性损失
            physics_loss = torch.tensor(0.0, device=device)
            if physics_loss_module is not None:
                # 获取embedding和物理特征
                lncrna_emb, protein_emb, lncrna_physics, protein_physics = model.get_embeddings_and_physics(batch_data)
                if lncrna_physics is not None and protein_physics is not None:
                    # 计算物理相似度矩阵
                    physics_matrices, _, _ = physics_loss_module(lncrna_emb, protein_emb, lncrna_physics,
                                                                 protein_physics)
                    # 计算物理损失
                    sigmoid_output = torch.sigmoid(output)
                    physics_loss = physics_loss_module.compute_physics_loss(sigmoid_output, physics_matrices)

            # 计算总损失
            total_loss = main_loss + physics_loss_weight * physics_loss
            total_loss_item += total_loss.item()

            # 收集所有预测和标签用于计算评估指标
            all_labels.append(labels)
            all_outputs.append(torch.sigmoid(output))  # 应用sigmoid获取概率

    # 计算评估指标
    all_labels = torch.cat(all_labels, dim=0)
    all_outputs = torch.cat(all_outputs, dim=0)

    from utils import metrics
    eval_metrics = metrics(all_labels, all_outputs)

    return total_loss_item / len(eval_batch_data), eval_metrics


def main():
    try:
        # 检查是否有通过命令行指定的参数文件
        if args.load_best_params:
            # 如果指定了最佳参数文件，则加载参数
            if load_best_parameters(args):
                print("使用加载的最佳参数进行训练")
            else:
                print("无法加载指定的最佳参数文件，使用默认参数进行训练")
        else:
            # 检查默认路径下是否存在best_hyperparameters.json文件
            default_param_file = r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI_PL_C\best_hyperparameters.json'
            if os.path.exists(default_param_file):
                args.load_best_params = default_param_file
                if load_best_parameters(args):
                    print("使用默认路径下的最佳参数进行训练")
                else:
                    print("无法加载默认路径下的最佳参数文件，使用默认参数进行训练")
            else:
                print("未找到最佳参数文件，使用默认参数进行训练")

        interaction_lable_data = pd.read_csv(
            r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI\dataset\data1\lncRNA_protein_interaction_matrix.csv',
            index_col=0, header=0)

        # 直接使用graph_generator.data，避免额外的变量赋值
        graph_generator = Generate_Heterogeneous_Graph(interaction_lable_data, args)
        # 获取异构图数据
        heterogeneous_graph_data = graph_generator.data
        # 确保异构图数据在正确的设备上
        heterogeneous_graph_data = heterogeneous_graph_data.to(device)

        # 调试图结构
        # debug_graph_structure(heterogeneous_graph_data)

        # 获取训练所需的 正负样本对的特征
        # aggregated_feature_data是返回的异构图，包括节点特征，边索引等内容
        aggregated_feature_data = load_and_aggregate_features(
            args,
            heterogeneous_graph_data,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,  # 使用从最佳参数加载的num_layers
            agg_type='gcn'
        )
        print('邻居节点信息聚合完成')

        # 获取训练所需的正负样本对以及标签
        # 先进行初始化，初始化的过程中会完成异构图的构建以及负样本的采样
        # 使用同一个数据加载器，但通过train参数区分训练集和测试集
        data_loader = Load_data(
            args.dataset,
            args.batch_size,
            args,
            aggregated_feature_data,
            train=True  # 这个参数现在用于区分训练/测试集
        )

        # 获取训练需要使用的批数据
        train_batch_data = data_loader.generate_train_data_batch()

        # 创建测试集数据加载器
        eval_data_loader = Load_data(
            args.dataset,
            args.batch_size,
            args,
            aggregated_feature_data,
            train=False
        )
        eval_batch_data = eval_data_loader.generate_train_data_batch()

        # 添加调试信息，检查训练集和测试集的数据分布
        print(f"训练集批次数量: {len(train_batch_data)}")
        print(f"测试集批次数量: {len(eval_batch_data)}")

        print('数据加载完成')
        # 初始化模型
        # 获取实际特征维度
        sample_batch = next(iter(train_batch_data))
        lncrna_dim = sample_batch['lncrna_features'].shape[1]
        protein_dim = sample_batch['protein_features'].shape[1]

        # 获取物理特征维度（如果存在）
        lncrna_physics_dim = sample_batch.get('lncrna_physics_features', torch.empty(0)).shape[1] if 'lncrna_physics_features' in sample_batch and sample_batch['lncrna_physics_features'].numel() > 0 else 0
        protein_physics_dim = sample_batch.get('protein_physics_features', torch.empty(0)).shape[1] if 'protein_physics_features' in sample_batch and sample_batch['protein_physics_features'].numel() > 0 else 0

        print(f"实际lncrna特征维度: {lncrna_dim}")
        print(f"实际protein特征维度: {protein_dim}")
        print(f"实际lncrna物理特征维度: {lncrna_physics_dim}")
        print(f"实际protein物理特征维度: {protein_physics_dim}")

        model = PLLPI_Physics(
            lncrna_dim=lncrna_dim,
            protein_dim=protein_dim,
            lncrna_physics_dim=lncrna_physics_dim,
            protein_physics_dim=protein_physics_dim,
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate
        ).to(device)

        # 初始化物理损失模块（如果提供了物理特征）
        physics_loss_module = None
        if lncrna_physics_dim > 0 and protein_physics_dim > 0:
            physics_loss_module = PhysicsLossCombined(
                embedding_dim=128,
                physics_feature_dim=max(lncrna_physics_dim, protein_physics_dim),
                combined_dim=128 + max(lncrna_physics_dim, protein_physics_dim),
                num_physics_types=1  # 修改为1，因为只使用疏水性特征
            ).to(device)
            print("已初始化物理损失模块")
        else:
            print("未提供物理特征，跳过物理损失计算")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # 添加学习率调度器，设置最小学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

        # 用于跟踪训练过程中的指标
        train_metrics_history = []
        eval_metrics_history = []
        train_losses = []
        eval_losses = []
        learning_rates = []  # 记录学习率变化

        for epoch in range(args.epochs_num):
            train_output, train_metrics = train(model, optimizer, physics_loss_module, train_batch_data, epoch,
                                                args.warmup_epochs, args.physics_loss_weight)
            eval_output, eval_metrics = eval(model, optimizer, physics_loss_module, eval_batch_data,
                                             args.physics_loss_weight)
            # 更新学习率
            scheduler.step(eval_output)

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # 如果学习率过低，给出警告
            if current_lr < 1e-6:
                print(f"警告: 学习率过低 ({current_lr:.2e})，可能影响模型继续学习")

            print(
                f"Epoch {epoch + 1}/{args.epochs_num}, LR: {current_lr:.6f}, Train Loss: {train_output:.4f}, Eval Loss: {eval_output:.4f}")
            print(
                f"  Train Metrics - Acc: {train_metrics['accuracy']:.4f}, Prec: {train_metrics['precision']:.4f}, Rec: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['auc']:.4f}, AUPR: {train_metrics['aupr']:.4f}")
            print(
                f"  Eval Metrics - Acc: {eval_metrics['accuracy']:.4f}, Prec: {eval_metrics['precision']:.4f}, Rec: {eval_metrics['recall']:.4f}, F1: {eval_metrics['f1']:.4f}, AUC: {eval_metrics['auc']:.4f}, AUPR: {eval_metrics['aupr']:.4f}")

            # 保存指标历史
            train_metrics_history.append(train_metrics)
            eval_metrics_history.append(eval_metrics)
            train_losses.append(train_output)
            eval_losses.append(eval_output)

        # 计算平均指标
        print("\n=== 平均指标 ===")
        avg_train_metrics = {}
        avg_eval_metrics = {}

        metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'aupr']
        for metric in metric_names:
            avg_train_metrics[metric] = sum([m[metric] for m in train_metrics_history]) / len(train_metrics_history)
            avg_eval_metrics[metric] = sum([m[metric] for m in eval_metrics_history]) / len(eval_metrics_history)

        print("训练集平均指标:")
        for metric, value in avg_train_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

        print("测试集平均指标:")
        for metric, value in avg_eval_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

        # 找出最好和最差的指标
        print("\n=== 最好和最差指标 ===")
        best_eval_epoch = max(range(len(eval_metrics_history)), key=lambda i: eval_metrics_history[i]['f1'])
        worst_eval_epoch = min(range(len(eval_metrics_history)), key=lambda i: eval_metrics_history[i]['f1'])

        print(f"最好指标 (Epoch {best_eval_epoch + 1}):")
        best_metrics = eval_metrics_history[best_eval_epoch]
        for metric, value in best_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

        print(f"最差指标 (Epoch {worst_eval_epoch + 1}):")
        worst_metrics = eval_metrics_history[worst_eval_epoch]
        for metric, value in worst_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

        # 绘制指标曲线图
        print("\n=== 绘制指标曲线 ===")
        plot_individual_metrics(train_metrics_history, eval_metrics_history, args.output_dir)
        plot_loss_curve(train_losses, eval_losses, args.output_dir)
        print("已完成所有指标图表的绘制")

    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()


if __name__ == '__main__':
    main()