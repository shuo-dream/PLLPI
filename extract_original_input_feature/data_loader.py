import pandas as pd
def load_data(file_path):
    data=pd.read_csv(file_path)
    return data

def get_sequence_data(file_path):
    data=load_data(file_path)
    # zip():将多个序列按位置配对组合,返回一个 zip 对象（迭代器）
    data=list(zip(data['id'],data['sequence']))
    # print(data)
    return data