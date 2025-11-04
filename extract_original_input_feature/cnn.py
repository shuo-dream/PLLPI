import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear=nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.linear(x)

class SequenceFeatureAlignment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # 1D卷积层提取局部特征
        # 这个不是图卷积，是一维卷积，所以有4个输入通道，每个通道都是一个一维序列，形状为 (sequence_length,)，整体输入形状: (batch_size, 4, sequence_length)
        '''
            # 这个不是图卷积，是一维卷积，所以有4个输入通道，每个通道都是一个一维序列，形状为 (sequence_length,)，整体输入形状: (batch_size, 4, sequence_length)
            对于 self.conv1 来说：
                卷积核数量: hidden_dim=64 个卷积核
                每个卷积核的形状: (input_dim, kernel_size) = (4, 3)
                这意味着每个卷积核会同时跨4个通道进行操作
        
            padding=1的作用:
                在序列两端各添加1个位置的零填充
                使得输出序列长度与输入序列长度保持一致
                
            输入形状: (batch_size, input_dim, sequence_length)
            卷积核设置: kernel_size=3, padding=1
            输出形状: (batch_size, hidden_dim, sequence_length)
            
            eg:
                输入形状: (batch_size, input_channels, sequence_length)
                         (1, 4, N)
                
                通道维度 (4个通道):
                通道0: [f0_pos1, f0_pos2, f0_pos3, ..., f0_posN]
                通道1: [f1_pos1, f1_pos2, f1_pos3, ..., f1_posN]
                通道2: [f2_pos1, f2_pos2, f2_pos3, ..., f2_posN]
                通道3: [f3_pos1, f3_pos2, f3_pos3, ..., f3_posN]
        
                最终输出:
                特征图0: [out0_pos0, out0_pos1, out0_pos2, ..., out0_posN]
                特征图1: [out1_pos0, out1_pos1, out1_pos2, ..., out1_posN]
                ...
                特征图63: [out63_pos0, out63_pos1, out63_pos2, ..., out63_posN]
        '''
        # 每个卷积层的权重都是独立随机初始化的    虽然参数配置相似，但实际权重值不同
        # 卷积核有自己的值，卷积核的值和被卷积的值相乘相加就是本次卷积的结果，也就是加权求和
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        # 自适应池化到固定长度
        # 将任意长度的序列自适应地池化到固定长度10
        '''
            eg:
                # 输入: (batch_size, 64, variable_length)
                # 输出: (batch_size, 64, 10)
                每个输出位置是原序列中多个位置的平均值
        '''
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)  # 池化到10个位置

        self.linear = Linear(
            input_dim=hidden_dim * 10,
            hidden_dim=256,
            output_dim=output_dim
        )

    def forward(self, x):
        # x shape: (batch_size, features, sequence_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # 自适应池化
        x = self.adaptive_pool(x)  # shape: (batch_size, hidden_dim, 10)
        # 展平并映射到最终维度
        '''
            x.size(0): 保持批次维度不变
            -1: 自动推断剩余维度大小，将后面所有维度合并
            
            eg:
                # 变换前: (32, 64, 10)  # 32个样本，64个通道，10个位置
                # 变换后: (32, 640)     # 32个样本，640个特征
            是将后面两个维度直接拼接起来了吗？
            比如说 特征图0: [out0_pos0, out0_pos1, out0_pos2, ..., out0_pos10] 
            特征图1: [out1_pos0, out1_pos1, out1_pos2, ..., out1_pos10] 
            ... 
            特征图63: [out63_pos0, out63_pos1, out63_pos2, ..., out63_pos10] 
            变成了[out0_pos0, out0_pos1, out0_pos2, ..., out0_pos10,out1_pos0, out1_pos1, out1_pos2, ..., out1_pos10,,out63_pos0, out63_pos1, out63_pos2, ..., out63_pos10]
        '''
        x = x.view(x.size(0), -1)
        # print('x.shape[0]:',x.shape[0])
        # print('x.shape[1]:',x.shape[1])
        x = self.linear(x)
        return x
