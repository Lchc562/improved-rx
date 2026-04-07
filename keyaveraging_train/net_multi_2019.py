import simon  # 直接调用运行了simon.py，里面直接运行的语句会被执行
import numpy as np
# from pympler import asizeof
from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制大部分 TensorFlow 的日志消息。这可以使您在使用 TensorFlow 时减少控制台输出的冗长性
bs = 16384
wdir = './freshly_trained_nets/'


def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    # print(res)
    return res


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return res


# make residual tower of convolutional blocks
def make_resnet(num_blocks=4, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=10,
                reg_param=0.0001, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size,))
    rs = Reshape((num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)
    # 改变张量的形状。它的参数是一个元组(2 * num_blocks, word_size)，指定了新的形状。这里将输入张量inp重新调整为形状为(2 * num_blocks, word_size)
    # 创建一个输入层。它的参数shape指定了输入数据的形状，这里是一个一维向量，长度为num_blocks * word_size * 2
    # 用于重新排列张量的维度顺序。它的参数是一个元组(2, 1)，表示重新排列后的维度顺序。这里将rs张量的维度重新排列为(word_size, 2 * num_blocks)

    # add a single residual layer that will expand the data to num_filters channels
    # this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)
    # add residual blocks
    shortcut = conv0
    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])  # 残差
    # add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    # model.summary()
    return model


def train_speck_distinguisher(num_train, num_eval, num_epochs, num_rounds=7, depth=1, batch_size=10 ** 5, num_block=4,
                              k=1, diff=(0x0, 0x6)):
    # create the network
    # print("1")
    with strategy.scope():
        net = make_resnet(num_blocks=num_block, depth=depth, reg_param=10 ** -5)
        # 编译模型：指定优化器，损失函数和评估指标
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    # generate training and validation data

    X, Y = simon.make_datas_demo(num_train, num_rounds, k=k, diff=diff)
    X_eval, Y_eval = simon.make_datas_demo(num_eval, num_rounds, k=k, diff=diff)
    # memory_size = asizeof.asizeof(X)/(1024*1024)
    # memory_size1 = asizeof.asizeof(X_eval)/(1024*1024)
    # print(memory_size,"heihie",memory_size1)
    # set up model checkpoint
    check = make_checkpoint(wdir + 'best' + str(num_rounds) + 'depth' + str(depth) + '.h5')
    # # create learnrate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    # train and evaluate
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size, validation_data=(X_eval, Y_eval), callbacks=[lr, check])
    net.save('k'+str(k)+'diff_'+str(diff[0])+'_'+str(diff[1])+'.h5')
    np.save(wdir + 'h' + str(num_rounds) + '@r_depth' + str(depth) + '_vali_acc.npy', h.history['val_acc'])
    np.save(wdir + 'h' + str(num_rounds) + '@r_depth' + str(depth) + '_vali_loss.npy', h.history['val_loss'])
    np.save(wdir + 'h' + str(num_rounds) + '@r_depth' + str(depth) + '_train_loss.npy', h.history['loss'])
    np.save(wdir + 'h' + str(num_rounds) + '@r_depth' + str(depth) + '_train_acc.npy', h.history['acc'])
    np.save(wdir + 'h' + str(num_rounds) + 'r_depth' + str(depth) + '_lr.npy', h.history['lr'])
    # print(h.history)
    dump(h.history, open(wdir + 'hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    return net, h


# 步骤1：查看GPU设备情况
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#
# # 步骤2: 创建MirroredStrategy，使用多GPU
# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
# 步骤3：分布式策略下创建模型，把模型放入多个GPU中
# train_speck_distinguisher(num_train=2 ** 26, num_eval=2 ** 20, num_epochs=100, num_rounds=11, depth=10, batch_size=2 ** 15,
#                           num_block=4, k=15, diff=(0x0, 0x0003))

# from keras.callbacks import Callback
# import numpy as np
#
# class LossCallback(Callback):
#     def __init__(self):
#         super(LossCallback, self).__init__()
#         self.losses = []
#
#     def on_train_batch_end(self, batch, logs=None):
#         self.losses.append(logs['loss'])
#
# model.fit(x_train, y_train, epochs=10, callbacks=[LossCallback()])
#
# # 保存损失值到文件
# np.save('losses.npy', np.array(losses))
