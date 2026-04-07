import keras
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
# import  keras.models
import simeck
import simon
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
# 加载已训练好的模型
model = load_model('13_12_14_14to14_simon32_data25_pairs28_4_22_r14_depth10_acc_0.524078369140625.h5') # 替换 'your_model.h5' 为你的模型文件路径
# model = load_model('16_17_simeck32_data27_1_04_pairs36_r17_depth10.h5')
# X, Y = simeck.make_train_data(2**18, 17, pairs=36, k=1, diff=(0x0,0x0004))
X, Y = simon.make_datas_pairs(2**18, 14, pairs=28, k=4, diff=(0x0,0x0022))

# 计算准确率
loss, acc = model.evaluate(X,Y,batch_size=2**15)
# accuracy = np.mean(np.argmax(Y) == np.argmax(y_pred))
print(f"准确率: {acc * 100:.2f}%")