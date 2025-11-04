import pandas as pd
import numpy as np
from cnn import SequenceFeatureAlignment
import torch

# Kyte-Doolittle疏水性 scales
HYDROPHOBICITY_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# 氨基酸的pKa值 (用于计算净电荷)
'''
    pKa是解离常数的负对数，表示分子或离子失去/获得质子的能力
    pKa值越小，表示该基团越容易失去质子(在较低pH下就解离)
    pKa值越大，表示该基团越难失去质子(在较高pH下才解离)
    
    1.酸性氨基酸
    'D': 3.9 (天冬氨酸, Aspartic acid)
    'E': 4.3 (谷氨酸, Glutamic acid)
    这些具有较低的 pKa 值，容易失去质子
    2.碱性氨基酸
    'H': 6.0 (组氨酸, Histidine)
    'K': 10.5 (赖氨酸, Lysine)
    'R': 12.5 (精氨酸, Arginine)
    这些具有较高的 pKa 值，容易接受质子
    3.中性氨基酸
    'C': 8.3 (半胱氨酸, Cysteine)
    'Y': 10.1 (酪氨酸, Tyrosine)
    这些的 pKa 值介于中间范围
    4.蛋白质末端基团
    'N_term': 8.0 (N端氨基)
    'C_term': 3.1 (C端羧基)
'''
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
            # HYDROPHOBICITY_SCALE 是事先定义好的每个氨基酸的疏水性，直接将蛋白质序列中的氨基酸替换成对应的疏水性值即可
            # 但是这样的话，序列的长度不同，提取的特征长度就不一样了
            if aa in HYDROPHOBICITY_SCALE:
                hydrophobicity_values.append(HYDROPHOBICITY_SCALE[aa])
            else:
                hydrophobicity_values.append(0)  # 对于未知氨基酸设为0

        # print(np.array(hydrophobicity_values))
        return np.array(hydrophobicity_values)

    def calculate_net_charge(self, sequence, pH=7.4):
        """根据给定pH值计算序列的净电荷"""
        charge = 0.0

        # N端电荷，指蛋白质链氨基端（N端）的电荷贡献
        # 第一个氨基酸中含有氨基，最后一个氨基酸中含有羧基
        if sequence:
            n_term_aa = sequence[0]
            if n_term_aa in ['H', 'K', 'R']:
                # 碱性氨基酸作为N端时带正电荷
                charge += 1.0
            else:
                # 其他氨基酸N端基于通用pKa
                charge += 1.0 / (1.0 + 10 ** (pH - PKA_VALUES['N_term']))

        # C端电荷，指蛋白质链羧基端（C端）的电荷贡献
        if sequence:
            c_term_aa = sequence[-1]
            # print('c_term_aa:', c_term_aa)
            # C端总是带负电荷
            charge -= 1.0 / (1.0 + 10 ** (PKA_VALUES['C_term'] - pH))

        # 侧链电荷，指氨基酸侧链基团的电荷贡献
        # N端和C端只是指氨基和羧基贡献的电荷，而计算侧链基团的电荷贡献时，还需要计算第一个和最后一个氨基酸的侧链基团的电荷贡献，所以不需要排除第一个和最后一个氨基酸
        # 内部氨基酸：氨基酸的α-氨基和α-羧基都参与了肽键的形成，失去了游离的氨基和羧基
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

    def extract_protein_features(self, data, save_path, pH=7.4):
        """
        只提取蛋白质的物理化学性质特征（疏水性、净电荷、侧链大小和极性）

        参数:
        data: 蛋白质序列数据列表，格式为[("protein_id", "sequence"), ...]
        save_path: 特征保存路径
        pH: 计算净电荷时使用的pH值，默认为7.4（生理pH）
        """
        # print('data:',data)
        print(f"Starting to process {len(data)} protein sequences...")

        # sequence_lens = []
        # for protein_id, sequence in data:
        #     print('protein_id_length:', len(sequence))
        #     sequence_lens.append(len(sequence))
        #
        # print(f"Sequence length statistics:")
        # print(f"  Minimum length: {min(sequence_lens)}")
        # print(f"  Maximum length: {max(sequence_lens)}")
        # print(f"  Average length: {np.mean(sequence_lens):.2f}")

        all_features = []  # 用于收集所有特征

        cnn_model=SequenceFeatureAlignment(
            input_dim=4,
            hidden_dim=64,
            output_dim=128
        )
        for i in range(len(data)):
            # 获取序列数据
            # print('i:',i)
            protein_id, sequence = data[i]
            # print('protein_id:',protein_id)
            # print('sequence:',sequence)

            # 计算各种物理化学特征
            # 除了net_charge都是生成和序列长度一样的特征数量，也就是说会导致特征数量不一致
            # 计算序列的疏水性特征
            hydrophobicity = self.calculate_hydrophobicity(sequence)
            # 根据给定pH值计算序列的净电荷
            net_charge = self.calculate_net_charge(sequence, pH)
            # 计算序列的侧链范德华体积
            volume = self.calculate_side_chain_volume(sequence)
            # 计算序列的极性特征
            polarity = self.calculate_polarity(sequence)

            # 将所有特征组合成一个向量
            seq_len = len(sequence)

            # 组合所有特征
            # hydrophobicity等本来就是一维数组
            combined_features = np.array([
                hydrophobicity,
                volume,
                polarity,
                np.full(seq_len, net_charge)  # 将净电荷复制到每个位置
            ])

            # print('combined_features:',combined_features.shape)
            '''
                先拼接再卷积，因为特征交互：卷积层可以学习不同物理化学特征间的相互关系
                局部模式：能够捕获氨基酸局部区域内多种特征的组合模式
                信息完整性：保持了每个位置上所有特征的完整信息
                计算效率：只需要一次卷积操作，计算更高效
            '''
            '''
                eg: feature_tensor: torch.Size([1, 4, 1101])
                    feature_tensor: torch.Size([1, 4, 741])
                    
                    feature_tensor: torch.Size([1, 4, 1101]) 表示的是PyTorch张量的维度信息，具体含义如下：
                    第一个维度 1：表示批次大小(batch_size)，这里处理的是单个蛋白质序列样本
                    第二个维度 4：表示特征通道数(channels)，对应4种物理化学特征：
                    疏水性特征(hydrophobicity)
                    侧链体积特征(volume)
                    极性特征(polarity)
                    净电荷特征(net_charge)
                    第三个维度 1101：表示序列长度(sequence_length)，即该蛋白质序列包含1101个氨基酸
                    这种维度格式 [batch_size, channels, sequence_length] 是 nn.Conv1d 卷积层的标准输入格式，便于对序列数据进行一维卷积操作。
            '''
            feature_tensor=torch.FloatTensor(combined_features).unsqueeze(0)
            # print('feature_tensor:',feature_tensor.shape)
            alignment_features = cnn_model(feature_tensor)
            all_features.append(alignment_features)

        # 在保存结果之前，将张量转换为numpy数组
        all_features_numpy = []
        for feature in all_features:
            if torch.is_tensor(feature):
                # 如果是PyTorch张量，先detach再转换为numpy，将卷积神经网络输出的特征向量转换为一维数组，便于保存到 CSV 文件中。
                # print('feature.detach().numpy():',feature.detach().numpy())
                # print('feature.detach().numpy().flatten():',feature.detach().numpy().flatten())
                # .flatten()的作用是二维数组变成一维数组
                all_features_numpy.append(feature.detach().numpy().flatten())
            else:
                all_features_numpy.append(feature.flatten())

        print("Saving results...")
        df = pd.DataFrame(all_features_numpy)
        df.to_csv(save_path, index=False)
        print("save successfully")


class extract_lncrna_feature():
    def __init__(self):
        """
        初始化lncRNA特征提取器
        """
        # 核苷酸的电荷值（用于估算静电势）
        '''
            字典定义了RNA分子中各组分的电荷值：
            RNA碱基电荷：
            'A' (腺嘌呤)：-0.5
            'U' (尿嘧啶)：-0.5
            'G' (鸟嘌呤)：-0.5
            'C' (胞嘧啶)：-0.5
            末端电荷：
            'N_term' (N端)：-1.0
            'C_term' (C端)：-1.0
        '''
        self.CHARGE_VALUES = {
            'A': -0.5, 'U': -0.5, 'G': -0.5, 'C': -0.5,
            'N_term': -1.0, 'C_term': -1.0
        }

        # 核苷酸的疏水性值
        self.HYDROPHOBICITY_NUCLEOTIDE = {
            'A': 0.3, 'U': 0.2, 'G': 0.4, 'C': 0.1
        }

        # 碱基堆积能 (单位: kcal/mol)
        # 来源于实验测量和理论计算
        self.BASE_STACKING_ENERGY = {
            'AA': -7.0, 'AU': -6.7, 'UA': -8.5, 'UU': -9.0,
            'AG': -8.0, 'AC': -8.5, 'UG': -9.0, 'UC': -9.5,
            'GA': -8.0, 'GU': -8.5, 'GG': -10.0, 'GC': -12.0,
            'CA': -9.0, 'CU': -9.5, 'CG': -13.0, 'CC': -11.0
        }

    def calculate_debye_length(self, salt_concentration, temperature=298.15):
        """
        计算完整的Debye长度

        参数:
        salt_concentration: 盐浓度 (M)
        temperature: 温度 (K)，默认25°C

        返回:
        kappa_inv: Debye长度 (nm)
        """
        # 物理常数
        epsilon_0 = 8.854187817e-12  # 真空介电常数 (F/m)
        epsilon_r = 78.54  # 水的相对介电常数
        k_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)
        e = 1.602176634e-19  # 元电荷 (C)
        N_A = 6.02214076e23  # 阿伏伽德罗常数

        # 计算Debye长度，Debye长度 λ_D = 1/κ
        # κ² = (2 * e² * I) / (ε₀ * εᵣ * k_B * T)
        # 其中 I 是离子强度 (mol/m³)

        # 将盐浓度从 M 转换为 mol/m³
        ionic_strength = salt_concentration * 1000  # mol/m³

        # 计算κ² = (2 * e² * I) / (ε₀ * εᵣ * k_B * T)中的k
        # np.sqrt():开根号
        kappa = np.sqrt((2 * e ** 2 * ionic_strength * N_A) /
                        (epsilon_0 * epsilon_r * k_B * temperature))

        # 转换为纳米单位
        # 计算Debye长度 λ_D = 1/κ
        # * 1e9 是为了进行单位转换。κ的单位：m⁻¹ (每米)，Debye长度的单位：m (米)，实际应用单位：nm (纳米)
        kappa_inv_nm = (1.0 / kappa) * 1e9

        return kappa_inv_nm

    def calculate_electrostatic_potential(self, sequence, salt_concentration=0.15, temperature=298.15):
        """
        改进的静电势计算模型

        参数:
        sequence: RNA序列
        salt_concentration: 盐浓度(M)，默认为0.15M生理盐浓度
        temperature: 温度(K)，默认298.15K(25°C)

        返回:
        electrostatic_potentials: 静电势数组
        """
        electrostatic_potentials = []

        # 使用完整Debye长度计算
        debye_length = self.calculate_debye_length(salt_concentration, temperature)

        # 物理常数
        epsilon_0 = 8.854187817e-12  # 真空介电常数 (F/m)
        epsilon_r = 78.54  # 水的相对介电常数
        e = 1.602176634e-19  # 元电荷 (C)

        for i, nt in enumerate(sequence):
            if nt in self.CHARGE_VALUES:
                charge = self.CHARGE_VALUES[nt]

                # 简化的距离因子（假设相邻核苷酸间距离约0.34nm）
                distance_nm = (i + 1) * 0.34  # nm
                distance_m = distance_nm * 1e-9  # 转换为米

                # 使用完整的Debye-Hückel方程形式
                # Ψ = (q / (4πε₀εᵣr)) * exp(-r/λ_D)
                if distance_m > 0:  # 避免除零错误
                    prefactor = (charge * e) / (4 * np.pi * epsilon_0 * epsilon_r * distance_m)
                    exponential_term = np.exp(-distance_m / (debye_length * 1e-9))  # 转换nm到m
                    electrostatic_potential = prefactor * exponential_term
                else:
                    electrostatic_potential = 0.0

                electrostatic_potentials.append(electrostatic_potential)
            else:
                electrostatic_potentials.append(0.0)

        return np.array(electrostatic_potentials)

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

    def calculate_base_stacking_energy(self, sequence):
        """
        计算碱基堆积能特征

        参数:
        sequence: RNA序列

        返回:
        stacking_energies: 碱基堆积能数组
        """
        stacking_energies = []

        for i in range(len(sequence)):
            if i < len(sequence) - 1:
                # 获取相邻两个碱基
                # sequence[i:i + 2]是左闭右开
                dinucleotide = sequence[i:i + 2]
                if dinucleotide in self.BASE_STACKING_ENERGY:
                    stacking_energies.append(self.BASE_STACKING_ENERGY[dinucleotide])
                else:
                    stacking_energies.append(-8.0)  # 默认值
            else:
                # 最后一个核苷酸，使用前一个堆积能值或默认值
                if stacking_energies:
                    stacking_energies.append(stacking_energies[-1])
                else:
                    stacking_energies.append(-8.0)

        return np.array(stacking_energies)

    def extract_lncrna_features(self, data, save_path):
        """
        提取lncRNA的物理化学性质特征

        参数:
        data: lncRNA序列数据列表，格式为[("lncrna_id", "sequence"), ...]
        save_path: 特征保存路径
        """
        print(f"Starting to process {len(data)} lncRNA sequences...")

        all_features = []
        cnn_model = SequenceFeatureAlignment(
            input_dim=3,
            hidden_dim=64,
            output_dim=128
        )
        for i in range(len(data)):
            lncrna_id, sequence = data[i]

            # 转换为大写以匹配字典键
            sequence = sequence.upper()

            # 计算各种物理化学特征
            # 估算序列的静电势特征
            electrostatic = self.calculate_electrostatic_potential(sequence)
            # 计算核苷酸的疏水性特征
            hydrophobicity = self.calculate_nucleotide_hydrophobicity(sequence)
            # 计算碱基堆积能特征
            stacking_energy = self.calculate_base_stacking_energy(sequence)

            combined_features = np.array([
                electrostatic,
                hydrophobicity,
                stacking_energy
            ])

            combined_features=torch.FloatTensor(combined_features).unsqueeze(0)
            alignment_features = cnn_model(combined_features)
            all_features.append(alignment_features.detach().numpy())

        # 在保存结果之前，将张量转换为numpy数组
        all_features_numpy = []
        for feature in all_features:
            if torch.is_tensor(feature):
                # 如果是PyTorch张量，先detach再转换为numpy，将卷积神经网络输出的特征向量转换为一维数组，便于保存到 CSV 文件中。
                # .flatten()的作用是二维数组变成一维数组
                all_features_numpy.append(feature.detach().numpy().flatten())
            else:
                all_features_numpy.append(feature.flatten())

        print("Saving results...")
        df = pd.DataFrame(all_features_numpy)
        df.to_csv(save_path, index=False)
        print("save successfully")

