import pandas as pd
import numpy as np
import torch

# Kyte-Doolittle疏水性 scales
HYDROPHOBICITY_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# 氨基酸的pKa值 (用于计算净电荷)
PKA_VALUES = {
    'D': 3.9, 'E': 4.3, 'H': 6.0, 'C': 8.3, 'Y': 10.1,
    'K': 10.5, 'R': 12.5, 'N_term': 8.0, 'C_term': 3.1
}

# 范德华体积 (Å³)
VAN_DER_WAALS_VOLUME = {
    'A': 67, 'R': 148, 'N': 96, 'D': 91, 'C': 86,
    'Q': 114, 'E': 109, 'G': 48, 'H': 118, 'I': 124,
    'L': 124, 'K': 135, 'M': 124, 'F': 135, 'P': 90,
    'S': 73, 'T': 93, 'W': 163, 'Y': 141, 'V': 105
}

# 极性分类 (1表示极性，0表示非极性)
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

    def extract_protein_features(self, data, save_path, pH=7.4):
        """
        只提取蛋白质的疏水性特征

        参数:
        data: 蛋白质序列数据列表，格式为[("protein_id", "sequence"), ...]
        save_path: 特征保存路径
        pH: 计算净电荷时使用的pH值，默认为7.4（生理pH）
        """
        print(f"Starting to process {len(data)} protein sequences...")

        # 用于收集所有特征
        rows = []

        for i in range(len(data)):
            protein_id, sequence = data[i]

            # 计算疏水性特征
            hydrophobicity = self.calculate_hydrophobicity(sequence)

            # 聚合特征：计算平均疏水性
            avg_hydrophobicity = np.mean(hydrophobicity)

            # 将聚合后的特征添加到rows列表中
            rows.append([protein_id, avg_hydrophobicity])

        # 创建列名
        columns = ['protein_name', 'hydrophobicity']

        # 创建DataFrame并保存
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(save_path, index=True, index_label=None)
        print("save successfully")


class extract_lncrna_feature():
    def __init__(self):
        """
        初始化lncRNA特征提取器
        """
        # 核苷酸的疏水性值
        self.HYDROPHOBICITY_NUCLEOTIDE = {
            'A': 0.3, 'U': 0.2, 'G': 0.4, 'C': 0.1
        }

    def calculate_nucleotide_hydrophobicity(self, sequence):
        """
        计算核苷酸的疏水性特征

        参数:
        sequence: RNA序列

        返回:
        hydrophobicity_values: 疏水性数组
        """
        hydrophobicity_values = []

        for nt in sequence:
            if nt in self.HYDROPHOBICITY_NUCLEOTIDE:
                hydrophobicity_values.append(self.HYDROPHOBICITY_NUCLEOTIDE[nt])
            else:
                hydrophobicity_values.append(0.0)  # 未知核苷酸设为0

        return np.array(hydrophobicity_values)

    def extract_lncrna_features(self, data, save_path):
        """
        只提取lncRNA的疏水性特征

        参数:
        data: lncRNA序列数据列表，格式为[("lncrna_id", "sequence"), ...]
        save_path: 特征保存路径
        """
        print(f"Starting to process {len(data)} lncRNA sequences...")

        # 用于收集所有特征
        rows = []

        for i in range(len(data)):
            lncrna_id, sequence = data[i]

            # 转换为大写以匹配字典键
            sequence = sequence.upper()

            # 计算核苷酸的疏水性特征
            hydrophobicity = self.calculate_nucleotide_hydrophobicity(sequence)

            # 聚合特征：计算平均疏水性
            avg_hydrophobicity = np.mean(hydrophobicity)

            # 将聚合后的特征添加到rows列表中
            rows.append([lncrna_id, avg_hydrophobicity])

        # 创建列名
        columns = ['lncRNA_name', 'hydrophobicity']

        # 创建DataFrame并保存
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(save_path, index=True, index_label=None)
        print("save successfully")


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def get_sequence_data(file_path):
    data = load_data(file_path)
    data = list(zip(data['id'], data['sequence']))
    return data


if __name__ == "__main__":
    protein_feature_save_path = r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI_PL_C\physics_loss\output\original_physics_feature\original_protein_feature.csv'
    protein_lncrna_save_path = r'E:\postgraduate\y2025\CWS\my\my_project\PLLPI_PL_C\physics_loss\output\original_physics_feature\original_lncrna_feature.csv'

    rna_data_path = r'E:\postgraduate\y2025\CWS\my\my_project\preprocess_data\preprocessed_data\from_ICMF-LPI\data1\rna_id_sequence.csv'
    protein_data_path = r'E:\postgraduate\y2025\CWS\my\my_project\preprocess_data\preprocessed_data\from_ICMF-LPI\data1\protein_id_sequence.csv'
    protein_data = get_sequence_data(protein_data_path)
    rna_data = get_sequence_data(rna_data_path)

    # 创建特征提取器实例
    protein_extractor = extract_protein_feature()
    lncrna_extractor = extract_lncrna_feature()
    # 提取特征
    protein_extractor.extract_protein_features(protein_data, protein_feature_save_path)
    lncrna_extractor.extract_lncrna_features(rna_data, protein_lncrna_save_path)
