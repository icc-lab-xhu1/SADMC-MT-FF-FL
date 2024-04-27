import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import network_fed as net
import random
import time
import matplotlib.pyplot as plt
import math
from sklearn.metrics import auc

list_names_1 = []
list_names_2 = []

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class LoadData(Dataset):
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.len = len(self.train_x)

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.len

def train_and_test(class_name, train_loader, test_loader, num_classes, length):
    epoches = 100
    lr = 0.001  #
    input_num = 1
    output_num = num_classes
    name = class_name.rstrip('.csv')
    list_names_1.append(name)
    print(name)
    loss_list = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = net.MLSTM_test(length, num_classes)
    model = net.BasicFCN(input_num, num_classes, length)
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    acc = 0
    TN = 0
    all_t = 0
    all_f = 0
    pre_t = 0
    pre_f = 0
    all_2 = 0
    all_3 = 0
    pre_2 = 0
    pre_3 = 0
    preright_2 = 0
    preright_3 = 0
    preright_t = 0
    preright_f = 0
    correct = 0
    total = 0
    model_Tstate = None
    y_true = []
    y_pre = []

    for epoch in range(epoches):
        epoch_loss_list = []
        for images, labels in train_loader:
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            images = images.float()
            output = model(images)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_list.append(loss.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                model.eval()
                images = images.to(device)
                labels = labels.to(device)
                images = images.float()
                all_t += (labels == 0).sum().item()  # 正常样本数
                all_f += (labels == 1).sum().item()  # 异常样本数
                output = model(images)
                values, predicte = torch.max(output, 1)
                total += labels.size(0)
                y_true.append(labels)
                y_pre.append(predicte)
                pre_f += (predicte == 1).sum().item()  # 预测出的异常样本数
                if num_classes == 4:
                    all_2 += (labels == 2).sum().item()  # 异常样本数
                    all_3 += (labels == 3).sum().item()  # 异常样本数
                    pre_2 += (predicte == 2).sum().item()  # 预测出的异常样本数
                    pre_3 += (predicte == 3).sum().item()  # 预测出的异常样本数
                    preright_2 += ((predicte == 2) and (predicte == labels)).sum().item()
                    preright_3 += ((predicte == 3) and (predicte == labels)).sum().item()
                if num_classes == 3:
                    all_2 += (labels == 2).sum().item()  # 异常样本数
                    pre_2 += (predicte == 2).sum().item()  # 预测出的异常样本数
                    preright_2 += ((predicte == 2) and (predicte == labels)).sum().item()
                preright_f += ((predicte == 1) and (predicte == labels)).sum().item()
                TN += ((predicte == 0) and (predicte == labels)).sum().item()
                correct += (predicte == labels).sum().item()
        if (correct / total) > acc:
            acc = correct / total
            # print("The {} accuracy of epoch {} TSC: {}%".format(class_name,epoch+1, 100 * correct / total))
            # torch.save(model.state_dict(), "FedTemp1/" + class_name + ".pkl")
            model_Tstate = model

        loss_list.append(np.mean(epoch_loss_list))

    # plt.plot(loss_list, label='loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Iter_num')
    # plt.legend(list_names_1)
    # plt.savefig('E:/Latex/FGCS/figs//Loss_LL.pdf',dpi=600)
    acc = correct / total  # 准确率
    pre = preright_f / pre_f  # 精确率
    recall = preright_f / all_f  # 召回率
    f1_score = 2 * pre * recall / (pre + recall)  # f1 score
    if num_classes == 4:
        pre2 = preright_2 / pre_2  # 精确率
        recall2 = preright_2 / all_2  # 召回率
        f1_score2 = 2 * pre2 * recall2 / (pre2 + recall2)  # f1 score
        pre3 = preright_3 / pre_3  # 精确率
        recall3 = preright_3 / all_3  # 召回率
        f1_score3 = 2 * pre3 * recall3 / (pre3 + recall3)  # f1 score
        print('precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(pre2, recall2, f1_score2))
        print('precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(pre3, recall3, f1_score3))
    if num_classes == 3:
        pre2 = preright_2 / pre_2  # 精确率
        recall2 = preright_2 / all_2  # 召回率
        f1_score2 = 2 * pre2 * recall2 / (pre2 + recall2)  # f1 score
        print('precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(pre2, recall2, f1_score2))
    TP = preright_f
    FP = pre_f - preright_f
    FN = all_f - preright_f
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    # roc_auc_tran = auc(FPR, TPR)
    MCC = ((TP*TN)-(FP*FN)) / math.sqrt((TP+FP)*(FN+TP)*(FN+TN)*(FP+TN))
    # print('MCC:{:.3f},AUC:{:.3f}'.format(MCC,roc_auc_tran))
    print('precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(pre, recall, f1_score))
    if num_classes == 4:
        micro_pre = (preright_f + preright_2 + preright_3) / (pre_f + pre_2 + pre_3)
        mirco_recall = (preright_f + preright_2 + preright_3) / (all_f + all_2 + all_3)
        micro = 2 * micro_pre * mirco_recall / (micro_pre + mirco_recall)
        macro_pre = (pre + pre2 + pre3) / 3
        macro_recall = (recall + recall2 + recall3) / 3
        macro = (f1_score + f1_score2 + f1_score3) / 3
        macro = 2 * macro_pre * macro_recall / (macro_pre + macro_recall)
        print('macro:{:.3f}, micro:{:.3f}'.format(macro, micro))
    if num_classes == 3:
        micro_pre = (preright_f + preright_2 ) / (pre_f + pre_2 )
        mirco_recall = (preright_f + preright_2) / (all_f + all_2 )
        micro = 2 * micro_pre * mirco_recall / (micro_pre + mirco_recall)
        macro_pre = (pre + pre2 ) / 2
        macro_recall = (recall + recall2) / 2
        # macro = (f1_score + f1_score2) / 2
        macro = 2 * macro_pre * macro_recall / (macro_pre + macro_recall)
        print('macro:{:.3f}, micro:{:.3f}'.format(macro, micro))
    return y_true, y_pre, model_Tstate


def train_and_test_load(class_name, train_loader, test_loader, num_classes, length, index, FCNmodel):
    epoches = 100  #
    lr = 0.001 #
    input_num = 1
    output_num = num_classes
    name = class_name.rstrip('.csv')
    list_names_2.append(name)
    print(name)
    loss_list = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    index = FCNmodel.StoreEW(index, top_k=2)

    # random_number = random.randint(0, 9)
    # print(random_number)
    # parameterlist = FCNmodel.LoadParameters(random_number)
    # parameterlist = FCNmodel.getAvgParameter(9)
    parameterlist = FCNmodel.LoadParameters(index[0])

    # model = net.OrdinaryTransferFCN(input_num, num_classes, length, parameterlist)
    model = net.MLSTM(length, num_classes, parameterlist)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr)

    acc = 0
    TN = 0
    all_t = 0
    all_f = 0
    pre_t = 0
    pre_f = 0
    all_2 = 0
    all_3 = 0
    pre_2 = 0
    pre_3 = 0
    preright_2 = 0
    preright_3 = 0
    correct = 0
    total = 0
    preright_t = 0
    preright_f = 0
    model_Rstate = None
    test_data = []
    y_true = []
    y_pre = []

    for epoch in range(epoches):
        epoch_loss_list = []
        for images, labels in train_loader:
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = loss_func(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_list.append(loss.cpu().detach().numpy())

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                all_t += (labels == 0).sum().item()
                all_f += (labels == 1).sum().item()
                pre_out = model(images)
                values, predicte = torch.max(pre_out, 1)
                pre_t += (predicte == 0).sum().item()
                pre_f += (predicte == 1).sum().item()
                y_true.append(labels)
                y_pre.append(predicte)
                test_data.append(pre_out)
                if num_classes == 4:
                    all_2 += (labels == 2).sum().item()  # 异常样本数
                    all_3 += (labels == 3).sum().item()  # 异常样本数
                    pre_2 += (predicte == 2).sum().item()  # 预测出的异常样本数
                    pre_3 += (predicte == 3).sum().item()  # 预测出的异常样本数
                    preright_2 += ((predicte == 2) and (predicte == labels)).sum().item()
                    preright_3 += ((predicte == 3) and (predicte == labels)).sum().item()
                if num_classes == 3:
                    all_2 += (labels == 2).sum().item()  # 异常样本数
                    pre_2 += (predicte == 2).sum().item()  # 预测出的异常样本数
                    preright_2 += ((predicte == 2) and (predicte == labels)).sum().item()
                TN += ((predicte == 0) and (predicte == labels)).sum().item()
                preright_f += ((predicte == 1) and (predicte == labels)).sum().item()
                total += labels.size(0)
                correct += (predicte == labels).sum().item()
        if (correct / total) > acc:
            acc = correct / total
            # print("The {} accuracy of epoch {} TSC: {}%".format(class_name,epoch+1, 100 * correct / total))
            # torch.save(model.state_dict(),"basicFCN/"+class_name+".pkl")
            model_Rstate = model

        loss_list.append(np.mean(epoch_loss_list))

    y_true = torch.cat(y_true, dim=0)
    y_pre = torch.cat(y_pre, dim=0)
    test = torch.cat(test_data, dim=0)
    # plt.plot(loss_list, label='loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Iter_num')
    # plt.legend(list_names_2)
    # plt.savefig('E:/Latex/FGCS/figs/Loss_TL.pdf', dpi=600)
    pre = preright_f / pre_f# 精确率
    recall = preright_f / all_f  # 召回率
    f1_score = 2 * pre * recall / (pre + recall)# f1 score
    if num_classes == 4:
        pre2 = preright_2 / pre_2  # 精确率
        recall2 = preright_2 / all_2  # 召回率
        f1_score2 = 2 * pre2 * recall2 / (pre2 + recall2)  # f1 score
        pre3 = preright_3 / pre_3  # 精确率
        recall3 = preright_3 / all_3  # 召回率
        f1_score3 = 2 * pre3 * recall3 / (pre3 + recall3)  # f1 score
        print('precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(pre2, recall2, f1_score2))
        print('precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(pre3, recall3, f1_score3))
    if num_classes == 3:
        pre2 = preright_2 / pre_2  # 精确率
        recall2 = preright_2 / all_2  # 召回率
        f1_score2 = 2 * pre2 * recall2 / (pre2 + recall2)  # f1 score
        print('precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(pre2, recall2, f1_score2))
    TP = preright_f
    FP = pre_f - preright_f
    FN = all_f - preright_f
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    # roc_auc_tran = auc(FPR, TPR)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FP) * (FN + TP) * (FN + TN) * (FP + TN))
    # print('MCC:{:.3f},AUC:{:.3f}'.format(MCC, roc_auc_tran))
    print('precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(pre, recall, f1_score))

    if num_classes == 4:
        micro_pre = (preright_f + preright_2 + preright_3) / (pre_f + pre_2 + pre_3)
        mirco_recall = (preright_f + preright_2 + preright_3) / (all_f + all_2 + all_3)
        micro = 2 * micro_pre * mirco_recall / (micro_pre + mirco_recall)
        macro_pre = (pre + pre2 + pre3) / 3
        macro_recall = (recall + recall2 + recall3) / 3
        macro = (f1_score + f1_score2 + f1_score3) / 3
        macro = 2 * macro_pre * macro_recall / (macro_pre + macro_recall)
        print('macro:{:.3f}, micro:{:.3f}'.format(macro, micro))
    if num_classes == 3:
        micro_pre = (preright_f + preright_2 ) / (pre_f + pre_2 )
        mirco_recall = (preright_f + preright_2) / (all_f + all_2 )
        micro = 2 * micro_pre * mirco_recall / (micro_pre + mirco_recall)
        macro_pre = (pre + pre2 ) / 2
        macro_recall = (recall + recall2) / 2
        macro = (f1_score + f1_score2) / 2
        macro = 2 * macro_pre * macro_recall / (macro_pre + macro_recall)
        print('macro:{:.3f}, micro:{:.3f}'.format(macro, micro))
    return y_true, y_pre, test, model_Rstate

dir_name = ['vehicle1.csv', 'vehicle2.csv', 'vehicle3.csv', 'vehicle3.csv', 'vehicle4.csv','vehicle5.csv','vehicle6.csv','vehicle7.csv',\
            'vehicle8.csv','vehicle9.csv','vehicle10.csv']
# dir_name = ['swat-2015-data.csv', 'machine-1-1.csv', 'machine-1-2.csv', 'machine-1-3.csv', 'machine-1-4.csv']
# dir_name = ['carts.csv','catalogue.csv','front-end.csv','orders.csv','payment.csv','shipping.csv','user.csv']
# dir_name = ['carts.csv','catalogue.csv','front-end.csv','orders.csv','payment.csv','shipping.csv','user.csv','file_1.csv','file_2.csv', 'food_map.csv', 'inside.csv', 'order_other.csv', 'train.csv']
# dir_name = ['file_1.csv','file_2.csv', 'food_map.csv', 'inside.csv', 'order_other.csv', 'train.csv']

setup_seed(123)
Fed_iteration = 4
names = dir_name
start_time = time.time()
numbers = len(names)

from openTSNE import TSNE
from tSNE import utils

tsne = TSNE(
    perplexity=550,
    metric="cosine",
    n_jobs=16,
    random_state=35,
    verbose=True,
)

for i in range(len(names)):
    classname = names[i]
    # x_train = np.load("../data/"+classname+"_Xtrain.npy")
    # y_train = np.load("../data/"+classname+"_Ytrain.npy")
    # x_test = np.load("../data/"+classname+"_Xtest.npy")
    # y_test = np.load("../data/"+classname+"_Ytest.npy")
    x_train = np.load("../Mul_Class_Data/" + classname + "_Xtrain.npy")
    y_train = np.load("../Mul_Class_Data/" + classname + "_Ytrain.npy")
    x_test = np.load("../Mul_Class_Data/" + classname + "_Xtest.npy")
    y_test = np.load("../Mul_Class_Data/" + classname + "_Ytest.npy")
    num_classes = y_test.shape[1]
    length = x_train.shape[1]
    y_test = np.argmax(y_test,axis=1)
    y_train =  np.argmax(y_train,axis=1)
    print(x_train.shape,y_train.shape)
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1])).astype(np.float32)
    x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1])).astype(np.float32)
    train_loader = LoadData(x_train,y_train)
    test_set = LoadData(x_test,y_test)
    train_loader = dataloader.DataLoader(dataset=train_loader,batch_size=128,shuffle=True)
    test_loader = dataloader.DataLoader(dataset=test_set,shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Load---------dataset:", classname)
    if i == 0:
        y_true, y_pre, model_state = train_and_test(classname, train_loader, test_loader, num_classes,
                                                    length)
        shared = net.StoreParameters()
        shared.Appended(model_state)
        # del model_state
    else:
        y_true, y_pre, model_stateL = train_and_test(classname, train_loader, test_loader, num_classes,
                                                     length)
        shared.Appended(model_stateL)
    print("Dataset:%s eslapsed %.5f mins" % (classname, (time.time() - start_time) / 60))
plt.cla()

for e in range(1, Fed_iteration):
    FCNmodel = shared
    print("Fed Task------------%d"%(e+1))
    for i in range(len(names)):
        print(len(names))
        classname = names[i]
        x_train = np.load("../Mul_Class_Data/" + classname + "_Xtrain.npy")
        y_train = np.load("../Mul_Class_Data/" + classname + "_Ytrain.npy")
        x_test = np.load("../Mul_Class_Data/" + classname + "_Xtest.npy")
        y_test = np.load("../Mul_Class_Data/" + classname + "_Ytest.npy")
        num_classes = y_test.shape[1]
        length = x_train.shape[1]
        ###
        y_test = np.argmax(y_test, axis=1)
        y_train = np.argmax(y_train, axis=1)
        x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1])).astype(np.float32)
        x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1])).astype(np.float32)
        train_loader = LoadData(x_train,y_train)
        test_set = LoadData(x_test,y_test)
        train_loader = dataloader.DataLoader(dataset=train_loader,batch_size=128,shuffle=True)
        test_loader = dataloader.DataLoader(dataset=test_set,shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Load---------dataset:",classname)
        if i == 0:
            y_true_load, y_pre_load, test, model_stateP = train_and_test_load(classname,train_loader,test_loader,num_classes, length, i, FCNmodel)
            shared = net.StoreParameters()
            shared.Appended(model_stateP)

            # data = np.array(test.cpu())
            # true = np.array(y_true_load.cpu())
            # data = np.nan_to_num(data)
            # true = np.nan_to_num(true)
            # embedding_train = tsne.fit(data)
            # utils.plot(embedding_train, true)
            # plt.savefig('E:/Latex/FGCS/figs/'+classname+'.pdf', dpi=600)
        else:
            y_true_load, y_pre_load, test, model_stateV = train_and_test_load(classname,train_loader,test_loader,num_classes, length, i, FCNmodel)
            shared.Appended(model_stateV)

            # data = np.array(test.cpu())
            # true = np.array(y_true_load.cpu())
            # data = np.nan_to_num(data)
            # true = np.nan_to_num(true)
            # embedding_train = tsne.fit(data)
            # utils.plot(embedding_train, true)
            # plt.savefig('E:/Latex/FGCS/figs/'+classname+'.pdf', dpi=600)
        print("Dataset:%s eslapsed %.5f mins" % (classname, (time.time() - start_time) / 60))

