import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import network as net
import random
import time

class BasicFCN(nn.Module):
    def __init__(self,input_dims,num_classes,length):
        super(BasicFCN,self).__init__()
        self.conv1 = nn.Conv1d(input_dims,128,9,stride=1,padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 =nn.Conv1d(128,256,5,1,padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256,128,5,1,2)
        self.bn3 = nn.BatchNorm1d(128)
        self.avg = nn.AdaptiveMaxPool1d(1)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(128,128)
        self.linear2 = nn.Linear(128,num_classes)
        # nn.init.kaiming_normal_(self.conv1.weight)
        # nn.init.kaiming_normal_(self.conv2.weight)
        # nn.init.kaiming_normal_(self.conv3.weight)
        # nn.init.kaiming_normal_(self.linear1.weight)
        # nn.init.kaiming_normal_(self.linear2.weight)
        # nn.init.constant_(self.bn1.weight,val=1)
        # nn.init.constant_(self.bn2.weight,1)
        # nn.init.constant_(self.bn3.weight,val=1)
        # nn.init.constant_(self.conv1.bias,val=0)
        # nn.init.constant_(self.bn1.bias,val=0)
        # nn.init.constant_(self.conv2.bias,val=0)
        # nn.init.constant_(self.bn2.bias,val=0)
        # nn.init.constant_(self.conv3.bias,val=0)
        # nn.init.constant_(self.bn3.bias,val=0)
        # nn.init.constant_(self.linear1.bias,val=0)
        # nn.init.constant_(self.linear2.bias,val=0)
    def forward(self,x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        print(bn1.shape)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        print(bn2.shape)
        relu2 = self.relu(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        print(bn3.shape)
        relu3 = self.relu(bn3)
        print(relu3.shape)
        avg = self.avg(relu3)
        print(avg.shape)
        ful1 = self.linear1(avg.view(avg.size(0),-1))
        # drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(ful1)
        return res


class CNN_LSTM(nn.Module):
    def __init__(self, input_dims, num_classes, length):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dims, 128, 9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, 5, 1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm1d(128)
        # self.avg = nn.MaxPool1d(kernel_size=5,stride=1,padding=2)
        self.avg = nn.AdaptiveMaxPool1d(1)
        # self.flatten = nn.Flatten()
        self.lstm = nn.GRU(128, 128, 2)
        self.relu = nn.PReLU()
        self.linear1 = nn.Linear(128, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        relu2 = self.relu(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3)
        relu3 = self.relu(bn3)
        print(relu3.shape)
        avg = self.avg(relu3)
        print(avg.shape)
        out, _ = self.lstm(avg.view(avg.size(0), -1))
        ful1 = self.linear1(out)
        # drop = nn.Dropout(p=0.5)(ful1)
        res = self.linear2(ful1)
        return res

x = torch.randn(1, 31)
model = nn.LSTM(31,62,1)

y,_ = model(x)
print(y.shape)