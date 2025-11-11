# 超参数优化说明

本文档介绍了如何使用Optuna对PLLPI_PL_A模型进行超参数优化。

## 文件说明

- `hyperparameter_optimization.py`: 超参数优化主脚本
- `main.py`: 支持加载最佳超参数的训练脚本
- `best_hyperparameters.json`: 优化后生成的最佳超参数文件

## 使用方法

### 1. 运行超参数优化

```bash
python hyperparameter_optimization.py
```

该脚本将使用Optuna进行超参数搜索，并将结果保存到以下文件：
- `best_hyperparameters.json`: 最佳超参数配置
- `hyperparameter_optimization_results.csv`: 所有试验的详细结果

### 2. 使用最佳超参数训练模型

```bash
python main.py --load_best_params best_hyperparameters.json
```

或者直接运行（脚本会自动检测是否存在best_hyperparameters.json文件）：

```bash
python main.py
```

## 优化的超参数

超参数优化搜索空间包括：

- 学习率 (lr): 1e-4 到 1e-2 (对数分布)
- 批处理大小 (batch_size): [32, 64, 128]
- 训练轮数 (epochs_num): 50 到 150
- 物理损失权重 (physics_loss_weight): 0.001 到 0.1 (对数分布)
- 隐藏层维度 (hidden_dim): [64, 128, 256]
- 卷积层数量 (num_layers): 2 到 6
- 注意力头数 (head_num): [4, 8]
- Dropout率 (dropout_rate): 0.2 到 0.5
- 权重衰减 (weight_decay): 1e-5 到 1e-3 (对数分布)

## 早停机制

优化过程中实现了早停机制，如果验证集F1分数在15个epoch内没有改善，则提前停止训练。

## 结果分析

优化完成后，可以查看以下文件分析结果：

1. `best_hyperparameters.json`: 最佳超参数配置和对应的F1分数
2. `hyperparameter_optimization_results.csv`: 所有试验的详细结果，可用于进一步分析