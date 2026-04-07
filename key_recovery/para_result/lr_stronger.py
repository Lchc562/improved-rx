import numpy as np
from pympler import asizeof
from pickle import dump
import keras
from keras.models import load_model
from keras.datasets import mnist
import simeck
from basic_train import simon
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
import os
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制大部分 TensorFlow 的日志消息。这可以使您在使用 TensorFlow 时减少控制台输出的冗长性
bs = 16384



def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    # print(res)
    return res


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return res


def train_speck_distinguisher(num_train, num_eval, num_epochs, num_rounds=7, depth=1, batch_size=10 ** 5, num_block=4,
                              k=1, diff=(0x0, 0x6),pairs=1):
    with strategy.scope():
        net = load_model('16_17_simeck32_data27_pairs36_1_04_r17_depth10_acc_0.5040轻微欠拟合.h5')
        custom_op = tf.keras.optimizers.Adam(learning_rate=0.00001)
        net.compile(optimizer=custom_op, loss='mse', metrics=['acc'])
        # net.compile(optimizer='adam', loss='mse', metrics=['acc'])

    X, Y = simeck.make_train_data(num_train, num_rounds, pairs=pairs, k=k, diff=diff,demo=0)
    X_eval, Y_eval = simeck.make_train_data(num_eval, num_rounds, pairs=pairs, k=k, diff=diff,demo=2)

    src = '16_17_17_simeck' + str(simeck.WORD_SIZE() * 2) + '_data27_1_04' + "_pairs" + str(pairs) + '_r' + str(
        num_rounds) + '_depth' + str(depth)
    check = make_checkpoint(src + '.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size, validation_data=(X_eval, Y_eval), callbacks=[check],
                shuffle=True)
    dst = src + "_acc_" + str(np.max(h.history['val_acc']))
    os.rename(src + '.h5', dst + '.h5')
    net.save('Final_'+dst+'.h5')
    # dump(h.history, open('hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    return net, h


# 步骤1：查看GPU设备情况
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
train_speck_distinguisher(num_train=2 ** 16, num_eval=2 ** 18, num_epochs=20, num_rounds=17, depth=10,
                          batch_size=2 ** 16, num_block=2, k=1, diff=(0x0, 0x0004), pairs=36)
