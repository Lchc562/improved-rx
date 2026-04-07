import keras
from keras.models import load_model
from keras.datasets import mnist
import numpy as np
# import  keras.models
from basic_train import simon
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')
# 加载已训练好的模型
model = load_model('simon32_data14_pairs12_3_102_r13_depth10_acc_0.6128196716308594.h5') # 替换 'your_model.h5' 为你的模型文件路径

# 准备测试数据（MNIST数据集）
# X, Y = simon.make_datas_demo(2**18, 12, k=15, diff=(0x0,0x0003))
X, Y = simon.make_datas_pairs(2**18, 13, pairs=12, k=3, diff=(0x0,0x0102))

# 计算准确率
loss, acc = model.evaluate(X,Y,batch_size=2**14)
# accuracy = np.mean(np.argmax(Y) == np.argmax(y_pred))
print(f"准确率: {acc * 100:.2f}%")