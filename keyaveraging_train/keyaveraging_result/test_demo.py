import keras
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# np.set_printoptions(threshold=np.inf)
# 从npz文件中重新载入这些变量
data = np.load('x_xf1_y_yf1_z_r12_array11.npz')
variables = data.files
# 打印变量名
print(variables)

x = data['x']
print('shape', x.shape)
x_f1 = data['x_f1']
y = data['y']
y_f1 = data['y_f1']
z = data['z']
z_int = np.round(z).astype(int)
y_f1_int = np.round(y_f1).astype(int)
res = (y == z_int)
res_f1 = (y_f1_int == y)
# print(z[y==1][:50])
# print(z[y==0][:50])
# print("var1 loaded:", x[:2], x_f1[:2])
# print("var2 loaded:", y)
# print("var3 loaded:", z, z_int)
# accuracy = np.mean(res[y==0])
accuracy = np.mean(res)
print("Teacher_Acc: ", accuracy)
accuracy = np.mean(res[y == 1])
print("Teacher_TPR: ", accuracy)
accuracy = np.mean(res[y == 0])
print("Teacher_TNR: ", accuracy)

accuracy = np.mean(res_f1)
print("Real_Acc: ", accuracy)
accuracy = np.mean(res_f1[y == 1])
print("Real_TPR: ", accuracy)
accuracy = np.mean(res_f1[y == 0])
print("Real_TNR: ", accuracy)
