# 使用Optuna进行超参数优化

本文档介绍了如何在PLLPI项目中使用Optuna进行超参数优化。

## 文件说明

1. `hyperparameter_optimization.py` - 超参数优化主脚本
2. `best_hyperparameters.json` - 优化后保存的最佳超参数文件
3. `hyperparameter_optimization_results.csv` - 所有试验结果的详细记录

## 使用方法

### 1. 安装依赖

确保已安装Optuna库：

```bash
pip install optuna
```

### 2. 运行超参数优化

执行以下命令开始超参数优化：

```bash
python hyperparameter_optimization.py
```

默认会进行50次试验，搜索空间包括：
- 学习率 (lr): 1e-5 到 1e-2 (对数分布)
- 批处理大小 (batch_size): [32, 64, 128]
- 训练轮数 (epochs_num): 50 到 200
- 隐藏层维度 (hidden_dim): [64, 128, 256]
- 卷积层数量 (num_layers): 2 到 6
- Dropout率 (dropout_rate): 0.1 到 0.5
- 权重衰减 (weight_decay): 1e-5 到 1e-3 (对数分布)

### 3. 使用最佳超参数训练模型

优化完成后，最佳超参数会保存在 `best_hyperparameters.json` 文件中。使用以下命令加载最佳参数进行训练：

```bash
python main.py --load_best_params best_hyperparameters.json
```

或者，如果文件在默认位置，可以直接运行：

```bash
python main.py
```

程序会自动检测并加载最佳参数文件。

## 自定义优化

如果需要调整优化参数，可以修改 `hyperparameter_optimization.py` 中的以下部分：

1. 搜索空间：在 `objective` 函数中调整 `trial.suggest_*` 方法的参数
2. 试验次数：在 `main` 函数中修改 `study.optimize(objective, n_trials=50)` 的 `n_trials` 参数
3. 优化方向：在 `main` 函数中修改 `optuna.create_study(direction='maximize', ...)` 的 `direction` 参数

## 结果分析

优化完成后，会生成两个文件：
1. `best_hyperparameters.json` - 包含最佳超参数和对应的F1分数
2. `hyperparameter_optimization_results.csv` - 包含所有试验的详细结果，可用于进一步分析

可以使用pandas等工具分析结果：

```python
import pandas as pd

df = pd.read_csv('hyperparameter_optimization_results.csv')
print(df.sort_values('value', ascending=False).head())
```