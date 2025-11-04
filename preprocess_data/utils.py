import csv
from dataclasses import field


def data_loader(file_path):
    data={}
    with open(file_path,'r',encoding='utf-8') as file:
        id=None
        sequence=[]
        for line in file:
            # 移除字符串开头和末的所有空白字符
            line=line.strip()
            if line.startswith('>'):
                if id:
                    data[id]=''.join(sequence)
                    sequence=[]
                # 去掉开头的>
                full_id = line[1:]
                # 用if语句即可完成，没必要重新新一个单独的函数
                # 有的id中可能包含|或者空格，所以需要处理，过滤掉冗余数据，只获取id
                if '|' in full_id:
                    id = full_id.split('|')[1]
                elif ' ' in full_id:
                    id = full_id.split(' ')[0]
                else:
                    id = full_id
            else:
                sequence.append(line)
        # 处理最后一个序列
        if id:
            data[id] = ''.join(sequence)
    return data

def data_save(save_path, data):
    with open(save_path, 'w', newline='' , encoding='utf-8') as file:
        # 定义csv列名
        fieldnames = ['id', 'sequence']
        # 使用 csv.DictWriter 创建 CSV 写入器
        writer=csv.DictWriter(file, fieldnames=fieldnames)
        # 写入表头
        writer.writeheader()
        for id,sequence in data.items():
            writer.writerow({'id': id, 'sequence': sequence})

def get_data_sequence(file_path,output_path):
    data = data_loader(file_path)
    data_save(output_path, data)


def get_data_label(data):
    pass