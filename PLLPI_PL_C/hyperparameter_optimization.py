import argparse
import torch
import optuna
from data_load import Load_data
from model.PLLPI import PLLPI_Physics
from utils import set_random_seeds, metrics
from model.Node_information_aggregation import load_and_aggregate_features
from model.Generate_Heterogeneous_Graph import Generate_Heterogeneous_Graph
from physics_loss.physics_loss import PhysicsLossCombined
import pandas as pd
import traceback
import json
import os


def objective(trial):
    """
    Optuna优化目标函数
    """
    # 设置随机种子
    set_random_seeds(42)

    # 定义超参数搜索空间 - 优化所有使用到的超参数
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    epochs_num = trial.suggest_int('epochs_num', 50, 200)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 8)  # 扩展到8，与main.py中默认值一致
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    physics_loss_weight = trial.suggest_float('physics_loss_weight', 0.01, 1.0, log=True)
    head_num = trial.suggest_categorical('head_num', [2, 4, 8, 16])  # 减少选项
    scheduler_factor = trial.suggest_float('scheduler_factor', 0.1, 0.9)  # 优化学习率调度器因子
    scheduler_patience = trial.suggest_int('scheduler_patience', 3, 10)  # 优化学习率调度器耐心值
    early_stopping_patience = trial.suggest_int('early_stopping_patience', 10, 20)  # 优化早停耐心值
    grad_clip_norm = trial.suggest_float('grad_clip_norm', 0.5, 5.0)  # 优化梯度裁剪范数
    warmup_epochs = trial.suggest_int('warmup_epochs', 1, 10)  # 优化warmup轮数
    alpha_physics = trial.suggest_float('alpha_physics', 0.1, 0.9)  # 优化物理损失模块的alpha参数
    agg_num_layers = trial.suggest_int('agg_num_layers', 1, 4)  # 优化聚合层数量
    deep_feature_num_layers = trial.suggest_int('deep_feature_num_layers', 4, 12)  # 优化DeepFeatureExtractor的卷积层数量

    # 设置设备
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    try:
        # 加载数据
        dataset_path = r'G:\shen_cong\my\my_project\my_original_dataset\dataset\datasets\dataset1.txt'

        interaction_lable_data = pd.read_csv(
            r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI\dataset\data1\lncRNA_protein_interaction_matrix.csv',
            index_col=0, header=0)

        # 创建参数对象
        class Args:
            def __init__(self):
                self.cuda = 0 if torch.cuda.is_available() else -1
                self.batch_size = batch_size
                self.epochs_num = epochs_num
                self.lr = lr
                self.dataset = dataset_path
                self.warmup_epochs = warmup_epochs
                self.grad_clip_norm = grad_clip_norm
                self.device = device
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.deep_feature_num_layers = deep_feature_num_layers

        args = Args()

        # 构建异构图
        graph_generator = Generate_Heterogeneous_Graph(interaction_lable_data, args)
        heterogeneous_graph_data = graph_generator.data.to(device)

        # 聚合特征
        aggregated_feature_data = load_and_aggregate_features(
            args,
            heterogeneous_graph_data,
            hidden_dim=args.hidden_dim,
            num_layers=agg_num_layers,
        )

        # 加载训练数据
        data_loader = Load_data(
            args.dataset,
            args.batch_size,
            args,
            aggregated_feature_data,
            train=True
        )
        train_batch_data = data_loader.generate_train_data_batch()

        # 加载验证数据
        eval_data_loader = Load_data(
            args.dataset,
            args.batch_size,
            args,
            aggregated_feature_data,
            train=False
        )
        eval_batch_data = eval_data_loader.generate_train_data_batch()

        # 获取实际的特征维度
        sample_batch = next(iter(train_batch_data))
        lncrna_dim = sample_batch['lncrna_features'].shape[1]
        protein_dim = sample_batch['protein_features'].shape[1]
        print(f"实际特征维度 - lncRNA: {lncrna_dim}, protein: {protein_dim}")

        # 初始化模型
        model = PLLPI_Physics(
            lncrna_dim=lncrna_dim,
            protein_dim=protein_dim,
            hidden_dim=args.hidden_dim,
            dropout_rate=dropout_rate
        ).to(device)

        # 动态设置head_num参数
        model.cross_attention.head_num = head_num
        model.cross_attention.head_dim = 128 // head_num
        # 确保特征维度可以被头数整除
        assert 128 % head_num == 0, "feature_dim必须能被head_num整除"

        # 初始化物理损失模块
        physics_loss_module = PhysicsLossCombined(
            embedding_dim=128,
            physics_feature_dim=7,  # 根据实际情况调整
            combined_dim=128 + 7,  # 根据实际情况调整
            num_physics_types=1,  # 修复：根据实际物理特征类型数量设置
            alpha=alpha_physics  # 添加alpha参数
        ).to(device)

        # 优化器和学习率调度器
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=1e-7
        )

        best_f1 = 0.0
        early_stopping_counter = 0

        # 训练循环
        for epoch in range(args.epochs_num):
            # 训练阶段
            model.train()
            train_loss = 0
            all_labels = []
            all_outputs = []

            for batch_idx, batch_data in enumerate(train_batch_data):
                labels = batch_data['label']
                optimizer.zero_grad()

                output = model(batch_data)

                # 计算主要损失
                main_loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels.float())

                # 计算物理一致性损失
                physics_loss = torch.tensor(0.0, device=device)
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

                train_loss += total_loss.item()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                optimizer.step()

                all_labels.append(labels)
                sigmoid_output = torch.sigmoid(output)
                all_outputs.append(sigmoid_output)

            all_labels = torch.cat(all_labels, dim=0)
            all_outputs = torch.cat(all_outputs, dim=0)
            train_metrics = metrics(all_labels, all_outputs)

            # 验证阶段
            model.eval()
            eval_loss = 0
            eval_labels = []
            eval_outputs = []

            with torch.no_grad():
                for batch_data in eval_batch_data:
                    labels = batch_data['label']
                    output = model(batch_data)

                    # 计算主要损失
                    main_loss = torch.nn.functional.binary_cross_entropy_with_logits(output, labels.float())

                    # 计算物理一致性损失
                    physics_loss = torch.tensor(0.0, device=device)
                    lncrna_emb, protein_emb, lncrna_physics, protein_physics = model.get_embeddings_and_physics(
                        batch_data)
                    if lncrna_physics is not None and protein_physics is not None:
                        # 计算物理相似度矩阵
                        physics_matrices, _, _ = physics_loss_module(lncrna_emb, protein_emb, lncrna_physics,
                                                                     protein_physics)
                        # 计算物理损失
                        sigmoid_output = torch.sigmoid(output)
                        physics_loss = physics_loss_module.compute_physics_loss(sigmoid_output, physics_matrices)

                    # 计算总损失
                    total_loss = main_loss + physics_loss_weight * physics_loss
                    eval_loss += total_loss.item()

                    eval_labels.append(labels)
                    eval_outputs.append(torch.sigmoid(output))

            eval_labels = torch.cat(eval_labels, dim=0)
            eval_outputs = torch.cat(eval_outputs, dim=0)
            eval_metrics = metrics(eval_labels, eval_outputs)

            # 更新学习率
            scheduler.step(eval_loss)

            # 早停机制
            if eval_metrics['f1'] > best_f1:
                best_f1 = eval_metrics['f1']
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print(f"Trial {trial.number}, Epoch {epoch}: Early stopping triggered")
                break

            # 每10个epoch报告一次
            if epoch % 10 == 0:
                print(f"Trial {trial.number}, Epoch {epoch}: Train Loss: {train_loss / len(train_batch_data):.4f}, "
                      f"Eval Loss: {eval_loss / len(eval_batch_data):.4f}, Eval F1: {eval_metrics['f1']:.4f}")

        # 返回最佳F1分数作为优化目标
        return best_f1

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        traceback.print_exc()
        return 0.0


def save_best_params(study, filename='best_hyperparameters.json'):
    """
    保存最佳超参数到JSON文件
    """
    best_params = study.best_params
    best_params['best_f1_score'] = study.best_value

    with open(filename, 'w') as f:
        json.dump(best_params, f, indent=4)

    print(f"最佳参数已保存到 {filename}")


def main():
    # 创建Optuna研究对象
    study = optuna.create_study(direction='maximize', study_name='PLLPI_Hyperparameter_Optimization')

    # 添加日志记录
    print("开始超参数优化...")
    print("搜索空间:")
    print("- 学习率 (lr): 1e-5 到 1e-2 (对数分布)")
    print("- 批处理大小 (batch_size): [32, 64, 128]")
    print("- 训练轮数 (epochs_num): 50 到 200")
    print("- 隐藏层维度 (hidden_dim): [64, 128, 256]")
    print("- 卷积层数量 (num_layers): 2 到 8")
    print("- Dropout率 (dropout_rate): 0.1 到 0.5")
    print("- 权重衰减 (weight_decay): 1e-5 到 1e-3 (对数分布)")
    print("- 物理损失权重 (physics_loss_weight): 0.01 到 1.0 (对数分布)")
    print("- 注意力头数 (head_num): [4, 8, 16]")
    print("- 学习率调度器因子 (scheduler_factor): 0.1 到 0.9")
    print("- 学习率调度器耐心值 (scheduler_patience): 3 到 10")
    print("- 早停耐心值 (early_stopping_patience): 10 到 20")
    print("- 梯度裁剪范数 (grad_clip_norm): 0.5 到 5.0")
    print("- Warmup轮数 (warmup_epochs): 1 到 10")
    print("- 物理损失alpha参数 (alpha_physics): 0.1 到 0.9")
    print("- 聚合层数量 (agg_num_layers): 1 到 4")
    print("- DeepFeatureExtractor卷积层数量 (deep_feature_num_layers): 4 到 12")

    # 开始优化
    study.optimize(objective, n_trials=50)

    # 输出最佳结果
    print("\n=== 最佳超参数 ===")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    print(f"\n最佳F1分数: {study.best_value:.4f}")

    # 保存结果到文件
    df = study.trials_dataframe()
    df.to_csv('hyperparameter_optimization_results.csv', index=False)
    print("\n优化结果已保存到 hyperparameter_optimization_results.csv")

    # 保存最佳参数
    save_best_params(study, 'best_hyperparameters.json')


if __name__ == '__main__':
    main()