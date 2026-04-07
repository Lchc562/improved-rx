import keras
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
# import  keras.models
from keyaveraging_train import simon
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
# 加载已训练好的模型
model = load_model('simon_keyav_32_data15_diff1_15_03_r12_depth5_acc_0.578125.h5') # 替换 'your_model.h5' 为你的模型文件路径

# 准备测试数据（MNIST数据集）
X, x_f1, Y = simon.make_datas_demo(2**16, 12, k=15, diff=(0x0,0x3))
X = simon.data14_generate(X)
print(X.shape,Y.shape)
# print(X)
# 计算准确率
loss, acc = model.evaluate(X,Y,batch_size=2**12)
# accuracy = np.mean(np.argmax(Y) == np.argmax(y_pred))
print(f"准确率: {acc * 100:.2f}%")