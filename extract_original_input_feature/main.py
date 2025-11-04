from data_loader import get_sequence_data
from extract_feature import extract_protein_feature,extract_lncrna_feature

def main():
    protein_feature_save_path=r'G:\shen_cong\my\my_project\extract_original_input_feature\save_feature\data1\original_protein_feature.csv'
    protein_lncrna_save_path=r'G:\shen_cong\my\my_project\extract_original_input_feature\save_feature\data1\original_lncrna_feature.csv'

    rna_data_path=r'G:\shen_cong\my\my_project\preprocess_data\preprocessed_data\from_ICMF-LPI\data1\rna_id_sequence.csv'
    protein_data_path=r'G:\shen_cong\my\my_project\preprocess_data\preprocessed_data\from_ICMF-LPI\data1\protein_id_sequence.csv'

    protein_data=get_sequence_data(protein_data_path)
    rna_data=get_sequence_data(rna_data_path)

    # 创建特征提取器实例
    protein_extractor = extract_protein_feature()
    lncrna_extractor = extract_lncrna_feature()
    # 提取特征
    protein_extractor.extract_protein_features(protein_data, protein_feature_save_path)
    lncrna_extractor.extract_lncrna_features(rna_data,protein_lncrna_save_path)


main()