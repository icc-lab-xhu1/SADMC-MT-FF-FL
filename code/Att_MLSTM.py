import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import network as net
import random
import time


class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, queries):
        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MLSTM(nn.Module):
    def __init__(self, ni, nf, fc_drop_p=0.3):
        super(MLSTM, self).__init__()

        self.fc_drop_p = fc_drop_p
        self.num_classes = nf
        self.num_lstm_layers = 2
        self.num_lstm_out = 128

        self.conv1_nf = 128
        self.conv2_nf = 256
        self.conv3_nf = 128

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.conv1 = nn.Conv1d(1, self.conv1_nf, kernel_size=7, stride=1, padding=2)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, kernel_size=3, stride=1, padding=2)

        self.bn1 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.ea = ExternalAttention(d_model=self.num_lstm_out, S=128)

        self.lstm = nn.LSTM(input_size=ni,
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        self.relu = nn.ReLU()
        self.fc = nn.Linear(self.conv3_nf + self.num_lstm_out, self.num_classes)
        self.convDrop = nn.Dropout(self.fc_drop_p)

    def forward(self, x):
        x1, (ht, ct) = self.lstm(x)
        x1 = self.ea(x1)
        x1 = x1[:, -1, :]

        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.fc(x_all)
        x_out = F.log_softmax(x_out, dim=1)

        return x_out

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
    epoches = 30
    lr = 0.02  #
    input_num = 1
    output_num = num_classes
    loss_list = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MLSTM(length, num_classes)
    model.to(device)
    #     loss_func = nn.NLLLoss()
    #     optimizer = optim.Adam(model.parameters(), lr=lr)  #

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_func = nn.NLLLoss()

    acc = 0
    all_t = 0
    all_f = 0
    pre_t = 0
    pre_f = 0
    preright_t = 0
    preright_f = 0
    correct = 0
    total = 0
    model_Tstate = None
    #     torch.backends.cudnn.enabled=False
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
                pre_t += (predicte == 0).sum().item()  # 预测出的正常样本数
                pre_f += (predicte == 1).sum().item()  # 预测出的异常样本数
                preright_t += ((predicte == 0) and (predicte == labels)).sum().item()
                preright_f += ((predicte == 1) and (predicte == labels)).sum().item()
                correct += (predicte == labels).sum().item()
        if (correct / total) > acc:
            acc = correct / total
            # print("The {} accuracy of epoch {} TSC: {}%".format(class_name,epoch+1, 100 * correct / total))
            torch.save(model.state_dict(), "FedTemp1/" + class_name + ".pkl")
            model_Tstate = model

        loss_list.append(np.mean(epoch_loss_list))
    acc = correct / total  # 准确率
    pre = preright_f / pre_f  # 精确率
    recall = preright_f / all_f  # 召回率
    f1_score = 2 * pre * recall / (pre + recall)  # f1 score
    return acc, pre, recall, f1_score, model_Tstate

# dir_name = ['swat-2015-data.csv', 'machine-1-1_test.csv','machine-1-2_test.csv','machine-1-3_test.csv','machine-1-4_test.csv',
#             'carts_cpu-hog.csv', 'carts_latency.csv','catalogue_latency-s.csv','orders_latency-s.csv']
dir_name = ['carts_cpu-hog.csv', 'carts_latency.csv','catalogue_latency-s.csv','orders_latency-s.csv']

setup_seed(123)
Fed_iteration = 10
names = dir_name
start_time = time.time()
numbers = len(names)

for i in range(len(names)):
    classname = names[i]
    x_train = np.load("./data/"+classname+"_Xtrain.npy")
    y_train = np.load("data/"+classname+"_Ytrain.npy")
    x_test = np.load("data/"+classname+"_Xtest.npy")
    y_test = np.load("data/"+classname+"_Ytest.npy")
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

    if i == 0:
        acc, pre, recall, f1_score, model_state = train_and_test(classname, train_loader, test_loader, num_classes,
                                                                 length)
        shared = net.StoreParameters()
        shared.Appended(model_state)
        # del model_state
    else:
        acc, pre, recall, f1_score, model_stateL = train_and_test(classname, train_loader, test_loader, num_classes,
                                                                  length)
        shared.Appended(model_stateL)

    print(classname+':'+'acc:{:.3f}, precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(acc, pre, recall, f1_score))

for e in range(1,Fed_iteration):
    FCNmodel = shared
    logTxt = "results.txt"
    f = open(logTxt, mode="a+")
    f.writelines("FedAVG Task------------" + str(e + 1) + "\n")
    f.close()
    print("Fed Task------------%d"%(e+1))
    for i in range(len(names)):
        logTxt = "results.txt"
        f = open(logTxt, mode="a+")
        classname = names[i]
        x_train = np.load("data/"+classname+"_Xtrain.npy")
        y_train = np.load("data/"+classname+"_Ytrain.npy")
        x_test = np.load("data/"+classname+"_Xtest.npy")
        y_test = np.load("data/"+classname+"_Ytest.npy")
        num_classes = y_test.shape[1]
        length = x_train.shape[1]
        ###
        y_test = np.argmax(y_test, axis=1)
        y_train =  np.argmax(y_train, axis=1)
        x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1])).astype(np.float32)
        x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1])).astype(np.float32)
        train_loader = LoadData(x_train,y_train)
        test_set = LoadData(x_test,y_test)
        train_loader = dataloader.DataLoader(dataset=train_loader,batch_size=x_train.shape[0]//4,shuffle=True)
        test_loader = dataloader.DataLoader(dataset=test_set,shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Load---------dataset:",classname)
        if i == 0:
            acc, pre, recall, f1_score, model_stateP = train_and_test_load(classname,train_loader,test_loader,num_classes, length, i, FCNmodel)
            shared = net.StoreParameters()
            shared.Appended(model_stateP)
        else:
            acc, pre, recall, f1_score, model_stateV = train_and_test_load(classname,train_loader,test_loader,num_classes, length, i, FCNmodel)
            shared.Appended(model_stateV)
        print(classname+':'+'acc:{:.3f}, precision:{:.3f}, recall:{:.3f}, f1_score:{:.3f}'.format(acc, pre, recall, f1_score))