import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# Kyte-Doolittle疏水性 scales
# 存储了氨基酸的Kyte-Doolittle疏水性指数，正值代表疏水性，负值代表亲水性。
HYDROPHOBICITY_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# 氨基酸的pKa值 (用于计算净电荷)
# 记录了部分氨基酸及N端、C端的pKa值，用于计算蛋白质在特定pH条件下的电荷状态。
PKA_VALUES = {
    'D': 3.9, 'E': 4.3, 'H': 6.0, 'C': 8.3, 'Y': 10.1,
    'K': 10.5, 'R': 12.5, 'N_term': 8.0, 'C_term': 3.1
}

# 范德华体积 (Å³)
# 包含了各氨基酸的范德华体积（单位为Å³），反映其空间大小。
VAN_DER_WAALS_VOLUME = {
    'A': 67, 'R': 148, 'N': 96, 'D': 91, 'C': 86,
    'Q': 114, 'E': 109, 'G': 48, 'H': 118, 'I': 124,
    'L': 124, 'K': 135, 'M': 124, 'F': 135, 'P': 90,
    'S': 73, 'T': 93, 'W': 163, 'Y': 141, 'V': 105
}

# 极性分类 (1表示极性，0表示非极性)
# 对氨基酸进行了极性分类，其中1表示极性氨基酸，0表示非极性氨基酸。
POLARITY = {
    'A': 0, 'R': 1, 'N': 1, 'D': 1, 'C': 0,
    'Q': 1, 'E': 1, 'G': 0, 'H': 1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 1, 'T': 1, 'W': 0, 'Y': 1, 'V': 0
}


class extract_protein_feature():
    def __init__(self):
        """
        初始化蛋白质特征提取器
        """
        pass

    def calculate_hydrophobicity(self, sequence):
        """计算序列的疏水性特征"""
        hydrophobicity_values = []
        for aa in sequence:
            if aa in HYDROPHOBICITY_SCALE:
                hydrophobicity_values.append(HYDROPHOBICITY_SCALE[aa])
            else:
                hydrophobicity_values.append(0)  # 对于未知氨基酸设为0
        return np.array(hydrophobicity_values)

    def calculate_net_charge(self, sequence, pH=7.4):
        """根据给定pH值计算序列的净电荷"""
        charge = 0.0

        # N端电荷
        if sequence:
            n_term_aa = sequence[0]
            if n_term_aa in ['H', 'K', 'R']:
                # 碱性氨基酸作为N端时带正电荷
                charge += 1.0
            else:
                # 其他氨基酸N端基于通用pKa
                charge += 1.0 / (1.0 + 10 ** (pH - PKA_VALUES['N_term']))

        # C端电荷
        if sequence:
            c_term_aa = sequence[-1]
            # C端总是带负电荷
            charge -= 1.0 / (1.0 + 10 ** (PKA_VALUES['C_term'] - pH))

        # 侧链电荷
        for aa in sequence:
            if aa in ['D', 'E']:  # 酸性氨基酸
                charge -= 1.0 / (1.0 + 10 ** (pH - PKA_VALUES[aa]))
            elif aa in ['K', 'R']:  # 碱性氨基酸
                charge += 1.0 / (1.0 + 10 ** (PKA_VALUES[aa] - pH))
            elif aa == 'H':  # 组氨酸
                charge += 1.0 / (1.0 + 10 ** (PKA_VALUES[aa] - pH))

        return charge

    def calculate_side_chain_volume(self, sequence):
        """计算序列的侧链范德华体积"""
        volumes = []
        for aa in sequence:
            if aa in VAN_DER_WAALS_VOLUME:
                volumes.append(VAN_DER_WAALS_VOLUME[aa])
            else:
                volumes.append(0)  # 对于未知氨基酸设为0
        return np.array(volumes)

    def calculate_polarity(self, sequence):
        """计算序列的极性特征"""
        polarities = []
        for aa in sequence:
            if aa in POLARITY:
                polarities.append(POLARITY[aa])
            else:
                polarities.append(0)  # 对于未知氨基酸设为非极性
        return np.array(polarities)

    def extract_features(self, data, save_path, pH=7.4):
        """
        只提取蛋白质的物理化学性质特征（疏水性、净电荷、侧链大小和极性）

        参数:
        data: 蛋白质序列数据列表，格式为[("protein_id", "sequence"), ...]
        save_path: 特征保存路径
        pH: 计算净电荷时使用的pH值，默认为7.4（生理pH）
        """
        print(f"Starting to process {len(data)} protein sequences...")

        all_features = []  # 用于收集所有特征

        # 使用 tqdm 显示详细进度
        start_time = time.time()
        for i in tqdm(range(len(data)), desc="Processing sequences"):
            # 获取序列数据
            protein_id, sequence = data[i]

            # 计算各种物理化学特征
            hydrophobicity = self.calculate_hydrophobicity(sequence)
            net_charge = self.calculate_net_charge(sequence, pH)
            volume = self.calculate_side_chain_volume(sequence)
            polarity = self.calculate_polarity(sequence)

            # 将所有特征组合成一个向量
            seq_len = len(sequence)

            # 组合所有特征
            combined_features = np.hstack([
                hydrophobicity,
                volume,
                polarity,
                np.full(seq_len, net_charge)  # 将净电荷复制到每个位置
            ])

            all_features.append(combined_features)

            # 显示中间进度信息
            current_processed = i + 1
            if current_processed % 10 == 0:
                elapsed_time = time.time() - start_time
                print(f"\nProcessed {current_processed}/{len(data)} sequences ({elapsed_time:.2f}s elapsed)")

        print("Saving results...")
        df = pd.DataFrame(all_features)
        df.to_csv(save_path, index=False)
        print(f"Completed! Total time: {time.time() - start_time:.2f}s")


def extract_rna_feature():
    pass
