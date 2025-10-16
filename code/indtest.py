
import pickle
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score, roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from torch.utils.data import Subset
import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import interp
import warnings
import pandas as pd
from transformers import AutoTokenizer,AutoModel
import csv
import SE_modul,infonce
from EMA import EMA
warnings.filterwarnings("ignore")


Amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWYX'
# binary encoding
def create_encode_dataset(filepath):
    data_list = []
    result_seq_datas = []
    result_seq_labels = []
    with open(filepath, encoding='utf-8') as f:

        for line in f.readlines():
            x_data_sequence, label = list(line.strip('\n').split(','))
            data_list.append((x_data_sequence, label))
    for data in data_list:
        code = []
        # seq_index=1
        result_seq_labels.append(int(data[1]))
        for seq in data[0]:
            one_code = []
            for amino_acid_index in Amino_acid_sequence:
                if amino_acid_index == seq:
                    flag = 1
                else:
                    flag = 0
                one_code.append(flag)
            code.extend(one_code)
        result_seq_datas.append(code)
    return np.array(result_seq_datas), np.array(result_seq_labels, dtype=np.int32)
def get_bina(sequence):
    AA = 'ARNDCQEGHILKMFPSTWYVX'
    encodings = []
    code=[]
    for aa in sequence:
        if aa == '-':
            code = code + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,1]
            continue
        for aa1 in AA:
            tag = 1 if aa == aa1 else 0
            code.append(tag)
    encodings.append(code)
    return encodings

# train_filepath= '../Datasets/train.csv'
# test_filepath= '../Datasets/ind_test.csv'
#
# train_dataset, train_labels = create_encode_dataset(train_filepath)
# print(train_dataset.shape)
# test_dataset, test_labels = create_encode_dataset(test_filepath)
# print(test_dataset.shape)

# train_data = pd.read_csv("cluster_data.csv")
# train_data.reset_index(inplace=True)
# train_seq= np.array(train_data.iloc[:, 34])
# new_data = []
# for row in train_seq:
#     new_data.append(get_bina(row))
# trainfea_data = np.array(new_data)
# trainfea_data = np.squeeze(trainfea_data, axis=1)
# train_lable = np.array(train_data.iloc[:, 33])
train_data = pd.read_csv("../Datasets/dataset/train_encoded.csv")
train_data.reset_index(inplace=True)
trainfea_data = np.array(train_data.iloc[:, 22:602])
train_seq = np.array(train_data.iloc[:, 0])
train_lable = np.array(train_data.iloc[:, 1])

# trainfea_data = trainfea_data.reshape(-1, 31, 83)
test_data = pd.read_csv("../Datasets/dataset/test_encoded.csv")
test_data.reset_index(inplace=True)
testfea_data = np.array(test_data.iloc[:, 22:602])
test_seq = np.array(test_data.iloc[:, 0])
test_lable = np.array(test_data.iloc[:, 1])


class MyDataset(Dataset):
    def __init__(self, datas, labels,seq):
        self.datas = datas
        self.labels = labels
        self.seq=seq
    def __getitem__(self, index):
        # x_data = np.array(self.datas[index]).astype('float32').reshape((29, 21))
        x_data = np.array(self.datas[index]).astype('float32')
        y_data = self.labels[index]
        seq = self.seq[index]
        seq_len = len(seq)
        seq = seq.replace('', ' ')
        encoding = tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            max_length=seq_len + 2,
            return_token_type_ids=True,
            # pad_to_max_length=True,

            return_attention_mask=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        sample = {
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),

        }

        return x_data, y_data,sample

    def __len__(self):
        return len(self.datas)

# train_set = MyDataset(train_dataset, train_labels)
# test_set = MyDataset(test_dataset, test_labels)
train_set = MyDataset(trainfea_data, train_lable,train_seq)
test_set = MyDataset(testfea_data, test_lable,test_seq)

PATH='../ESM2-150M'
# AutoModel.from_pretrained(esm_model_path, torch_dtype=torch.float, trust_remote_code=True, add_pooling_layer=True)
tokenizer = AutoTokenizer.from_pretrained("../ESM2-150M", trust_remote_code=True)


class Model_LSTM_MutilHeadSelfAttention(nn.Module):

    def __init__(self,input_size, hidden_size, num_classes=2, num_layers=1):
        super(Model_LSTM_MutilHeadSelfAttention, self).__init__()
        self.input_size = input_size
        # hidden_size：
        self.hidden_size = hidden_size
        # num_classes：
        self.num_classes = num_classes
        # LSTM layers：
        self.num_layers = num_layers

        # BiLSTM Layer：
        self.Bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        # attention layer：
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2,num_heads=8,batch_first=True,dropout=0.5)
        self.dropout1 = nn.Dropout(0.9)

    def forward(self, inputs):
        input_ids = inputs
        # LSTM layer
        Bilstm_outputs, (last_hidden_state, last_cell_state) = self.Bilstm(inputs)
        Bilstm_outputs = self.dropout1(Bilstm_outputs)
        context,_ = self.attention(Bilstm_outputs,Bilstm_outputs,Bilstm_outputs)
        MutilHead_output = context
        # print("context shape:",context.shape)
        return (Bilstm_outputs, MutilHead_output), Bilstm_outputs


# input_size = len(Amino_acid_sequence)
input_size = 21
hidden_size = 64
num_classes = 2
num_layers = 1

class KcrNet(nn.Module):

    def __init__(self, input_classes=21, nums_classes=2,input_size=21,hidden_size=64,num_layers=1):
        super(KcrNet, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        # self.conv1 = torch.nn.Conv1d(in_channels=input_classes, out_channels=32, kernel_size=5, padding=2, stride=1)
        # self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)
        # self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=29, kernel_size=5, padding=2, stride=2)
        self.conv1 = torch.nn.Conv1d(in_channels=21, out_channels=32, kernel_size=5, padding=2, stride=1)
        self.conv2 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, padding=2, stride=2)
        self.conv3 = torch.nn.Conv1d(in_channels=32, out_channels=29, kernel_size=5, padding=2, stride=2)
        # self.BiLSTM_ATT=Model_LSTM_MutilHeadSelfAttention(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers)
        self.BiLSTM_ATT = Model_LSTM_MutilHeadSelfAttention(input_size=20, hidden_size=self.hidden_size,
                                                            num_layers=self.num_layers)
        self.attention = nn.MultiheadAttention(embed_dim=1792, num_heads=8, batch_first=True, dropout=0.5)

        self.ESM = AutoModel.from_pretrained("../ESM2-150M")
        # flatten layer
        self.Flatten = torch.nn.Flatten()
        # lienar layer
        # self.Linear1 = torch.nn.Linear(in_features=29 * 167, out_features=128)
        self.Linear1 = torch.nn.Linear(in_features=4352, out_features=128)
        # self.Linear1 = torch.nn.Linear(in_features=29 * 136, out_features=128)
        self.Linear2 = torch.nn.Linear(in_features=128, out_features=nums_classes)
        self.dropout1=torch.nn.Dropout(0.3)
        self.dropout2 = torch.nn.Dropout(0.3)

        self.Linear3=torch.nn.Linear(640,1024)
        self.Linear4 = torch.nn.Linear(1024, 128)
        self.Linear5 = torch.nn.Linear(512, 256)
        self.Linear6 = torch.nn.Linear(256, 128)
        self.Relu=torch.nn.ReLU()
        self.dropout3=torch.nn.Dropout(0.2)
        self.Linear7 = torch.nn.Linear(3968, out_features=nums_classes)

        self.Linear8 = torch.nn.Linear(640, 29)

        self.dropoutone = nn.Dropout(0.15)
        self.dropouttwo = nn.Dropout(0.3)
        self.dropoutthree = nn.Dropout(0.9)
        self.infonce_loss = infonce.InfoNCE()

        # 注意力
        self.w_omega = nn.Parameter(torch.Tensor(136, 64))
        self.u_omega = nn.Parameter(torch.Tensor(64, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
    def attention_net(self, x):
        u = torch.tanh(torch.matmul(x, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = torch.nn.functional.softmax(att, dim=1)
        scored_x = x * att_score
        # context = torch.sum(scored_x, dim=1)

        return scored_x

    def forward(self, x,input_ids,attention_mask):
        with torch.no_grad():
            pooled_output,sss = self.ESM(input_ids=input_ids,attention_mask=attention_mask,return_dict=False)
        # pooled_output = pooled_output.reshape(-1, 31, 32, 20)
        # resnetout = self.ResNet(pooled_output)

        # sss=sss.reshape(-1,32,20)
        # x=sss

        # x=torch.cat([x,sss],dim=-1)
        x=x.reshape(-1,29,20)




        inputs=x
        # x = torch.permute(x, [0, 2, 1]) #permute
        # # first conv1d
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.dropout1(x)
        # First_outputs=x
        #
        # #seconde conv1d
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # Second_outputs=x
        #
        # #third conv1d
        # x = self.conv3(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # Third_outputs=x

        #the outpur of BiLSTM and attention layers
        visual_outputs,BiLSTM_outputs=self.BiLSTM_ATT(inputs)

        #concate:
        # total_outputs=torch.cat([resnetout,BiLSTM_outputs],dim=-1)


        # resnetout=self.Flatten(resnetout)

        BiLSTM_outputs=self.Flatten(BiLSTM_outputs)
        x=self.Flatten(x)

        # total_outputs = torch.cat([resnetout, x], dim=-1)
        # x = self.Flatten(total_outputs)
        # context, _ = self.attention(resnetout, resnetout, resnetout)
        total_outputs = torch.cat([sss, BiLSTM_outputs], dim=-1)

        x = self.Linear1(total_outputs)
        # x = self.Linear1(sss)
        x = F.relu(x)  #activate funcation
        Linear_output=x
        x = self.Linear2(x)





        # total=torch.cat([x,esm4],dim=-1)

        # hh=torch.argmax(total)

        # x1,_=torch.max(x,dim=-1,keepdim=True)
        # x2,_=torch.max(esm4,dim=-1,keepdim=True)
        # total=torch.cat([x1,x2],dim=-1)



        # return (inputs,First_outputs,Second_outputs,Third_outputs,visual_outputs,Linear_output),x
        return ( x), x
def Calculate_confusion_matrix(y_test_true,y_pred_label,y_score):

    conf_matrix = confusion_matrix(y_test_true, y_pred_label)
    TN = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    AUROC = roc_auc_score(y_test_true, y_score)
    AUPRC = average_precision_score(y_test_true, y_score)

    return (TN,TP,FN,FP),(SN,SP,ACC,MCC,F1Score,AUROC,AUPRC)

def Calculate_Kfold_mean_std_metrics_values(total_SN,total_SP,total_ACC,total_F1_score,total_MCC,total_AUC,total_AUP):
    #calculate mean
    mean_SN = np.mean(total_SN)
    mean_SP = np.mean(total_SP)
    mean_ACC = np.mean(total_ACC)
    mean_F1_score = np.mean(total_F1_score)
    mean_MCC = np.mean(total_MCC)
    mean_AUC = np.mean(total_AUC)
    mean_AUP = np.mean(total_AUP)

    std_SN = np.std(total_SN)
    std_SP = np.std(total_SP)
    std_ACC = np.std(total_ACC)
    std_F1_score = np.std(total_F1_score)
    std_MCC = np.std(total_MCC)
    std_AUC = np.std(total_AUC)
    std_AUP = np.std(total_AUP)
    #
    # kfold_mean_metrics = []
    # kfold_mean_metrics.append(mean_SN)
    # kfold_mean_metrics.append(mean_SP)
    # kfold_mean_metrics.append(mean_ACC)
    # kfold_mean_metrics.append(mean_F1_score)
    # kfold_mean_metrics.append(mean_MCC)
    # kfold_mean_metrics.append(mean_AUC)
    #
    # kfold_std_metrics= []
    # kfold_std_metrics.append(std_SN)
    # kfold_std_metrics.append(std_SP)
    # kfold_std_metrics.append(std_ACC)
    # kfold_std_metrics.append(std_F1_score)
    # kfold_std_metrics.append(std_MCC)
    # kfold_std_metrics.append(std_AUC)
    print("5Kfold_Valid_Mean_metrics : SN is {:.3f},SP is {:.3f},ACC is {:.3f},F1-score is {:.3f},MCC is {:.3f},AUC is {:.3f},AUP is {:.3f}".
        format(mean_SN, mean_SP, mean_ACC, mean_F1_score, mean_MCC, mean_AUC,mean_AUP))
    print("5Kfold_Valid_Std_metrics : SN is {:.4f},SP is {:.4f},ACC is {:.4f},F1-score is {:.4f},MCC is {:.4f},AUC is {:.4f},AUP is {:.4f}".
          format(std_SN, std_SP, std_ACC, std_F1_score, std_MCC, std_AUC,std_AUP))

def cross_validation_train(model,epochs,train_loader,optimizer,train_criterion,device):
    print("train is start!")
    model.train()
    epoch_loss = []
    epoch_acc = []
    epoch_auc = []
    for epoch in range(epochs):
        model.train()
        for batch_id, data in enumerate(train_loader):

            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            input_ids = data[2]['input_ids'].to(device)
            attention_mask=data[2]['attention_mask'].to(device)
            _,y_predict = model(x_data,input_ids,attention_mask)
            loss = train_criterion(y_predict,y_data)

            aa=torch.argmax(y_predict, dim=1)
            for index, a in enumerate(aa):
                if a == 2:
                    aa[index] = 0
                elif a == 3:
                    aa[index] = 1
                else:
                    continue
            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                             aa.detach().cpu().numpy())

            auc = metrics.roc_auc_score(y_data.detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            epoch_loss.append(loss.detach().cpu().numpy())
            epoch_acc.append(acc)
            epoch_auc.append(auc)

            ema = EMA(model, decay=0.95).to(device)
            if batch_id % 10 == 0:
                ema.update(model)

            if (batch_id % 64 == 0):
                print("epoch is {},batch_id is {},loss is {},acc is:{},auc is:{}".format(epoch+1,batch_id,
                                                                                              loss.detach().cpu().numpy(),
                                                                                              acc, auc))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()


def cross_validate_test(model, valid_loader, criterion, device):

    model.eval()
    with torch.no_grad():

        valid_acc = []
        valid_loss = []
        valid_auc = []
        valid_auprc= []

        y_true = []
        y_score = []
        y_predict_labels_list = []

        for batch_id, data in enumerate(valid_loader):
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            input_ids = data[2]['input_ids'].to(device)
            attention_mask = data[2]['attention_mask'].to(device)
            _,y_predict = model(x_data,input_ids,attention_mask)
            aa = torch.argmax(y_predict, dim=1)
            for index, a in enumerate(aa):
                if a == 2:
                    aa[index] = 0
                elif a == 3:
                    aa[index] = 1
                else:
                    continue
            # y_predict_label = torch.argmax(y_predict, dim=1)
            # y_predict_labels_list.append(y_predict_label.detach().cpu().numpy())
            y_predict_label =aa
            y_predict_labels_list.append(y_predict_label.detach().cpu().numpy())
            loss = criterion(y_predict, y_data)
            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         aa.detach().cpu().numpy())

            auc = roc_auc_score(y_data[:].detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())
            # auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_data[:].detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            valid_loss.append(loss.detach().cpu().numpy())
            valid_acc.append(acc)
            valid_auc.append(auc)
            valid_auprc.append(auprc)
            y_true.append(y_data[:].detach().cpu().numpy())
            y_score.append(y_predict[:, 1].detach().cpu().numpy())

            if (batch_id % 64 == 0):
                print("batch_id is {},loss is {},acc is:{}, auc is {}".format(batch_id, loss.detach().cpu().numpy(),
                                                                              acc, auc))



        # concate:
        y_test_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        y_pred_label = np.concatenate(y_predict_labels_list)



        # confusion matrix
        (TN,TP,FN,FP),(SN,SP,ACC,MCC,F1Score,AUROC,AUPRC) = Calculate_confusion_matrix(y_test_true,y_pred_label,y_score)

        print('-----------------------------valid---------------------------------------------------------')
        print("Valid TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("Valid : SN is {},SP is {},ACC is {},F1-score is {},MCC is {},AUC is {},auprc is {}".
            format(SN, SP, ACC, F1Score, MCC, AUROC,AUPRC))


        valid_total_SN.append(SN)
        valid_total_SP.append(SP)
        valid_total_ACC.append(ACC)
        valid_total_F1_score.append(F1Score)
        valid_total_MCC.append(MCC)
        valid_total_AUC.append(AUROC) #roc_auc_area
        valid_total_AUP.append(AUPRC)
def cross_validate_test1(model, valid_loader, criterion, device):

    model.eval()
    with torch.no_grad():

        valid_acc = []
        valid_loss = []
        valid_auc = []
        valid_auprc = []

        y_true = []
        y_score = []
        y_predict_labels_list = []

        for batch_id, data in enumerate(valid_loader):
            x_data = data[0].to(device)
            y_data = data[1].to(device)
            y_data = torch.tensor(y_data, dtype=torch.long)
            input_ids = data[2]['input_ids'].to(device)
            attention_mask = data[2]['attention_mask'].to(device)
            _,y_predict= model(x_data,input_ids,attention_mask)
            aa = torch.argmax(y_predict, dim=1)
            for index, a in enumerate(aa):
                if a == 2:
                    aa[index] = 0
                elif a == 3:
                    aa[index] = 1
                else:
                    continue
            y_predict_label = aa
            y_predict_labels_list.append(y_predict_label.detach().cpu().numpy())
            loss = criterion(y_predict, y_data)
            acc = metrics.accuracy_score(y_data.detach().cpu().numpy(),
                                         aa.detach().cpu().numpy())

            auc = roc_auc_score(y_data[:].detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())
            auprc = average_precision_score(y_data[:].detach().cpu().numpy(), y_predict[:, 1].detach().cpu().numpy())

            valid_loss.append(loss.detach().cpu().numpy())
            valid_acc.append(acc)
            valid_auc.append(auc)
            valid_auprc.append(auprc)
            y_true.append(y_data[:].detach().cpu().numpy())
            y_score.append(y_predict[:, 1].detach().cpu().numpy())

            if (batch_id % 64 == 0):
                print("batch_id is {},loss is {},acc is:{}, auc is {}".format(batch_id, loss.detach().cpu().numpy(),
                                                                              acc, auc))


        # concate:
        y_test_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        y_pred_label = np.concatenate(y_predict_labels_list)


        # confusion matrix
        (TN,TP,FN,FP),(SN,SP,ACC,MCC,F1Score, AUROC,AUPRC) = Calculate_confusion_matrix(y_test_true,y_pred_label,y_score)

        print('-----------------------------test---------------------------------------------------------')
        print("test TP is {},FP is {},TN is {},FN is {}".format(TP, FP, TN, FN))
        print("test : SN is {},SP is {},ACC is {},F1-score is {},MCC is {},AUC is {},auprc is {}")

        test_total_SN.append(SN)
        test_total_SP.append(SP)
        test_total_ACC.append(ACC)
        test_total_F1_score.append(F1Score)
        test_total_MCC.append(MCC)
        test_total_AUC.append(AUROC) #roc_auc_area
        test_total_AUP.append(AUPRC)
def cross_validation_main(epochs,train_criterion,criterion):


    for run_idx  in range(5):
        torch.cuda.manual_seed_all(1800+run_idx*100)
        model = KcrNet(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        model.to(device)
        # print(model)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learn_rate)
        print(f"第{run_idx}次独立测试")
        batch_size = 256
        train_dataset = train_set

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=False)
        # training:
        cross_validation_train(model, epochs, train_loader, optimizer, train_criterion,  device)
        # # test:
        # model_path = '../results/modweights/'+str(fold) + '_DeepNhKcr_kfold_model.pth'.format(fold)
        # model.load_state_dict(
        #     torch.load(model_path, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")))

        print("eval is start>>>>>>>>>>>>>>>>>>>test>>>>>>>>>>>>>>>>>>>>>>>")
        cross_validate_test1(model, test_loader, criterion, device)



    print("最后结果>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    Calculate_Kfold_mean_std_metrics_values(test_total_SN, test_total_SP, test_total_ACC, test_total_F1_score, test_total_MCC, test_total_AUC,test_total_AUP)

def Kf_AUROC_show(plt, base_fpr, roc_auc, roc_auc_area):

    plt.figure(dpi=600)
    for i, item in enumerate(roc_auc):
        fpr, tpr = item
        plt.plot(fpr, tpr, label="ROC fold {} (AUC={:.1%})".format(i + 1, roc_auc_area[i]), lw=1, alpha=0.3)

    #calculate mean value
    plt.plot(base_fpr, np.average(tprs, axis=0),
             label=r'Mean ROC (AUC={:.1%} $\pm$ {:.2%})'.format (np.mean(roc_auc_area), np.std(roc_auc_area)),
             lw=1, alpha=0.8, color='b')
    #base line
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, alpha=0.8, color='c')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.savefig('../results/figures/DeepNhKcr-5Kfold.jpg')
    plt.show()

class FocalLoss(nn.Module):

    def __init__(self, alpha,gamma,reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction=reduction
    def forward(self, input, target):
        # input:size is N * 2. N　is the batch　size,
        # target:size is N. N is the batch size

        #claculate passibility:
        eps=1e-7
        pt = torch.softmax(input, dim=1)
        # passibility:
        # p1 = pt[:, 1]
        # p2 = pt[:, 3]
        p=pt[:, 1]
        loss = -self.alpha* torch.pow((1-p),self.gamma) * (target * torch.log(p + eps)) - \
               (1 - self.alpha) * torch.pow(p,self.gamma) * ((1 - target) * torch.log(1 - p + eps))
        if self.reduction == 'sum':
            loss=loss.sum() # sum
        else:
            loss=loss.mean() # mean
        return loss
class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=1, gamma_neg=5, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

if __name__ == '__main__':
    # # 设置随机种子
    SEED = 1800
    # torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # np.random.seed(SEED)
    # random.seed(SEED)
    # # 固定cuda的随机数种子，每次返回的卷积算法将是确定的
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True
    # # 模型架构保持不变、输入大小保持不变，启用该项可以提速
    # torch.backends.cudnn.benchmark = True

    learn_rate = 0.0001
    epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_criterion =FocalLoss(alpha=0.7,gamma=0.4)
    # train_criterion =ASLSingleLabel()
    criterion = nn.CrossEntropyLoss()

    #save metrics
    valid_total_SN = []
    valid_total_SP = []
    valid_total_ACC = []
    valid_total_F1_score = []
    valid_total_MCC = []
    valid_total_AUC = []
    valid_total_AUP = []

    test_total_SN = []
    test_total_SP = []
    test_total_ACC = []
    test_total_F1_score = []
    test_total_MCC = []
    test_total_AUC = []
    test_total_AUP = []

    roc_auc = []
    roc_auc_area = []
    tprs = []
    fprs = []
    base_fpr = np.linspace(0, 1, 101)
    base_fpr[-1] = 1.0

    # main function:
    cross_validation_main(epochs, train_criterion, criterion,)




