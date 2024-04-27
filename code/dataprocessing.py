import keras.utils as utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('E:/JupyterNotebook/SMD/smd/machine-1-1/machine-1-1_test.csv')
df = df.sample(10000)
print(df.head())

# 转换
# df["y"] = df["Normal/Attack"].replace(to_replace=["Normal", "Attack", "A ttack"], value=[0, 1, 1])
# df = df.drop(columns=["Normal/Attack"], axis=1)
df = df.drop(columns=["timestamp"], axis=1)
# df.drop(' Timestamp', axis=1, inplace=True)
print(df.head())

from sklearn import preprocessing
Y = df.iloc[:,-1]
X = df.iloc[:,0:-1]
Y = np.array(Y)
X = np.array(X)
processor = preprocessing.MinMaxScaler()
# processor = preprocessing.StandardScaler()
X = processor.fit_transform(X)
if np.isnan(X).any():
    X[np.isnan(X)] = np.nanmean(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
class_name = 'smd.csv'
num_classes = len(np.unique(y_train))
print(num_classes)
Y_train = utils.np_utils.to_categorical(y_train, num_classes)
Y_test = utils.np_utils.to_categorical(y_test, num_classes)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape, num_classes)
np.save("../vaedata/"+class_name+"_Xtrain",x_train)
np.save("../vaedata/"+class_name+"_Ytrain",Y_train)
np.save("../vaedata/"+class_name+"_Xtest",x_test)
np.save("../vaedata/"+class_name+"_Ytest",Y_test)
