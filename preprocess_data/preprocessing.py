import utils


def main():
    rna_output_path=r'G:\shen_cong\my\my_project\preprocess_data\preprocessed_data\from_ICMF-LPI\data1\rna_id_sequence.csv'
    protein_output_path=r'G:\shen_cong\my\my_project\preprocess_data\preprocessed_data\from_ICMF-LPI\data1\protein_id_sequence.csv'
    # 标签数据不用处理了
    lable_output_path=r'G:\shen_cong\my\my_project\preprocess_data\preprocessed_data\from_ICMF-LPI\data1\interaction_label.csv'

    rna_original_data_path_for_sequence = r"G:\shen_cong\my\my_project\my_original_dataset\origin_data_from_ICMF-LPI\data1\lncRNA.fasta"
    protein_original_data_path_for_sequence = r"G:\shen_cong\my\my_project\my_original_dataset\origin_data_from_ICMF-LPI\data1\protein.fasta"
    # original_data_path_for_interaction_label = r"G:\shen_cong\my\my_project\my_original_dataset\origin_data_from_ICMF-LPI\data1\lncRNA_protein_interaction_matrix.csv"

    get_rna_data_sequence=utils.get_data_sequence(rna_original_data_path_for_sequence,rna_output_path)
    get_protein_data_sequence=utils.get_data_sequence(protein_original_data_path_for_sequence,protein_output_path)
    # get_interaction_label_data=utils.get_data_label(original_data_path_for_interaction_label,lable_output_path)


main()