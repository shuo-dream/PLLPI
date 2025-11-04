import argparse
from operator import index
import pandas as pd
from tqdm import tqdm
import torch
import os
from data_load import Load_data
from model.PLLPI import PLLPI
from utils import set_random_seeds, debug_graph_structure, metrics
from model.Node_information_aggregation import load_and_aggregate_features
from model.Generate_Heterogeneous_Graph import Generate_Heterogeneous_Graph
# 导入绘图工具
from plot_utils import plot_individual_metrics, plot_loss_curve

# 创建参数解析器
parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs_num', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0005)  # 降低学习率以解决极端输出问题
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--dataset', type=str,
                    default=r'G:\shen_cong\my\my_project\my_original_dataset\dataset\datasets\dataset1.txt')
parser.add_argument('--output_dir', type=str, default='output', help='输出目录，用于保存图表等')
import traceback

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


def train(model, optimizer, train_batch_data):
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

        # 计算总损失
        total_loss = main_loss
        # 计算总损失
        train_loss += total_loss.item()

        # 反向传播
        total_loss.backward()  # 移除retain_graph=True
        # 梯度裁剪，防止梯度爆炸
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def eval(model, optimizer, eval_batch_data):
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

            # 计算总损失
            total_loss = main_loss
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
            hidden_dim=128,
            num_layers=2,
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
        model = PLLPI(
            lncrna_dim=128,
            protein_dim=128
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # 添加学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # 用于跟踪训练过程中的指标
        train_metrics_history = []
        eval_metrics_history = []
        train_losses = []
        eval_losses = []

        for epoch in range(args.epochs_num):
            train_output, train_metrics = train(model, optimizer, train_batch_data)
            eval_output, eval_metrics = eval(model, optimizer, eval_batch_data)
            # 更新学习率
            scheduler.step(eval_output)
            print(f"Epoch {epoch + 1}/{args.epochs_num}, Train Loss: {train_output:.4f}, Eval Loss: {eval_output:.4f}")
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