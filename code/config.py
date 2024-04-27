import os
import numpy as np
################# Root Files
data_files = 'E:/JupyterNotebook/DataSet/UCRArchive_2018'

dir_name = [ 'machine-1-1_test.csv','machine-1-2_test.csv','machine-1-3_test.csv']


New_dir = "./data/"
#_Xtrain.npy
#_Ytrain.npy

expand = 1
sub_expand = 1
loc = 0.05
scale = 0.05
locslist = [0.05,0.06,0.07,0.08,0.09,0.10,0.20]
scalelist = [0.05,0.06,0.07,0.08,0.09,0.10,0.20]
locslist = [i*expand for i in locslist]
scalelist = [i*sub_expand for i in scalelist]
per_class = [1,2,5,10,20,100,200]