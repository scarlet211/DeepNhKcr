import pandas as pd
from transformers import AutoTokenizer,AutoModel
from torch.utils.data import Dataset
import numpy as np
import csv

# class MyDataset(Dataset):
#     def __init__(self, datas, labels,seq):
#         self.datas = datas
#         self.labels = labels
#         self.seq=seq
#         self.csv_file=open("esm_encoded1.csv",'w')
#         self.csv_writer = csv.writer(self.csv_file)
#
#
#     def __getitem__(self, index):
#         y_data = self.labels[index]
#         csv_writer=self.csv_writer
#
#         seq = self.seq[index]
#         seq_len = len(seq)
#         seq = seq.replace('', ' ')
#         encoding = tokenizer.encode_plus(
#             seq,
#             add_special_tokens=True,
#             max_length=seq_len + 2,
#             return_token_type_ids=True,
#             pad_to_max_length=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#             padding='max_length',
#             truncation=True,
#         )
#
#         sample = {
#             'input_ids': encoding['input_ids'].flatten(),
#             'token_type_ids': encoding['token_type_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#
#         }
#         row = [seq, y_data] +sample['input_ids'].tolist()
#         csv_writer.writerow(row)
#
#
#         return  y_data,sample
#
#     def __len__(self):
#         return len(self.datas)
#
#     def __del__(self):
#         self.csv_file.close()
train_data = pd.read_csv("../Datasets/train_encoded.csv")
train_data.reset_index(inplace=True)
train_seq= np.array(train_data.iloc[:, 0])
train_lable = np.array(train_data.iloc[:, 1])
test_data = pd.read_csv("../Datasets/test_encoded.csv")
test_data.reset_index(inplace=True)
test_seq= np.array(test_data.iloc[:, 0])
test_lable = np.array(test_data.iloc[:, 1])
seq= np.concatenate((train_seq, test_seq), axis=0)
lable=np.concatenate((train_lable, test_lable), axis=0)


PATH='../ESM2-150M'
# AutoModel.from_pretrained(esm_model_path, torch_dtype=torch.float, trust_remote_code=True, add_pooling_layer=True)
tokenizer = AutoTokenizer.from_pretrained("../ESM2-150M", trust_remote_code=True)
# train_set = MyDataset(trainfea_data, train_lable,train_seq)
# print(train_set.__len__())
# csv_writer = csv.writer(open("esm_encoded.csv", 'w'))
# for
class MyDataset(Dataset):
    def __init__(self, seq_list, labels, file_path):
        self.seq_list = seq_list
        self.labels = labels
        self.file_path = file_path
        self.csv_file = open(file_path, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['sequence', 'label', 'input_ids'])

    def __getitem__(self, index):
        sequence = self.seq_list[index]
        label = self.labels[index]

        encoding = tokenizer(sequence, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].flatten()
        # token_type_ids = encoding['token_type_ids'].flatten()
        # attention_mask = encoding['attention_mask'].flatten()

        row = [sequence, label]+input_ids.tolist()
        self.csv_writer.writerow(row)

        return label, sequence

    def __len__(self):
        return len(self.seq_list)

    def __del__(self):
        self.csv_file.close()



# 创建数据集实例并存储编码结果到CSV文件
dataset = MyDataset(train_seq, train_lable, "encoded_sequences.csv")
for i in range(len(dataset)):
    label, sequence = dataset[i]

print("编码结果已存储到 encoded_sequences.csv 文件中。")