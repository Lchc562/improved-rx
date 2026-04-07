import keras
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
# import  keras.models
from simeck import simeck
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
# 加载已训练好的模型
model = load_model('14_13_15_15to15_simeck32_data14_diff1_15_03_r15_depth10_pairs8_acc_0.5093536376953125.h5') # 替换 'your_model.h5' 为你的模型文件路径

X, Y = simeck.make_train_data(2**16, 15, pairs=8, k=15, diff=(0x0,0x0003))

# 计算准确率
loss, acc = model.evaluate(X,Y,batch_size=2**12)
# accuracy = np.mean(np.argmax(Y) == np.argmax(y_pred))
print(f"准确率: {acc * 100:.2f}%")
