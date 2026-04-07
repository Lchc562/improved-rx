import simeck  # 直接调用运行了simon.py，里面直接运行的语句会被执行
import numpy as np
import gc
from pympler import asizeof
from pickle import dump
from tensorflow.keras.backend import concatenate
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
wdir = './simeck_basic_result/'


def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs - 1) - i % num_epochs) / (num_epochs - 1) * (high_lr - low_lr)
    # print(res)
    return res


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only=True)
    return res


def make_resnet1(pairs=1, num_blocks=4, num_filters=32, num_outputs=1, d1=512, d2=64, word_size=16, ks=3, depth=5,
                 reg_param=0.0001, final_activation='sigmoid'):
    inp = Input(shape=(int(num_blocks * word_size * pairs),))
    perm = Reshape((pairs, num_blocks * word_size))(inp)
    # perm = Permute((1, 3, 2))(rs)

    conv01 = Conv1D(num_filters, kernel_size=1, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    conv02 = Conv1D(num_filters, kernel_size=5, padding='same',
                    kernel_regularizer=l2(reg_param))(perm)
    c2 = concatenate([conv01, conv02], axis=-1)
    conv0 = BatchNormalization()(c2)
    conv0 = Activation('relu')(conv0)
    shortcut = conv0

    for i in range(depth):
        conv1 = Conv1D(num_filters * 2, kernel_size=ks, padding='same',
                       kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters * 2, kernel_size=ks,
                       padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])
        ks += 2

    dense0 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(dense0)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return (model)


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


def train_speck_distinguisher(num_train, num_eval, num_epochs, pairs=1, num_rounds=7, depth=1, batch_size=10 ** 5,
                              num_block=4,k=1, diff=(0x0, 0x6)):
    print(k,diff,hex(diff[1]))
    X, Y = simeck.make_train_data(num_train, num_rounds, pairs=pairs, k=k, diff=diff)
    print("train_set_over\n")
    X_eval, Y_eval = simeck.make_train_data(num_eval, num_rounds, pairs=pairs, k=k, diff=diff)
    print("vali_set_over\n")
    with strategy.scope():
        # net = make_resnet(num_blocks=num_block, depth=depth, reg_param=10 ** -5)
        net = make_resnet1(pairs=pairs, num_blocks=num_block, depth=depth, reg_param=10 ** -5)
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    src = wdir + 'simeck' + str(simeck.WORD_SIZE() * 2) + '_data27_pairs42_' + str(k) + '_' + str(hex(diff[1])[2:]) \
          + '_r' + str(num_rounds) + '_depth' + str(depth)
    # src = wdir + 'simeck' + str(simeck.WORD_SIZE() * 2) + '_data14_diff1_1_02' + '_r' + str(
    #     num_rounds) + '_depth' + str(depth) + "_pairs" + str(pairs)
    check = make_checkpoint(src + '.h5')
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size, validation_data=(X_eval, Y_eval), callbacks=[lr, check],
                shuffle=True)
    del X,Y,X_eval,Y_eval
    gc.collect()
    # dump(h.history, open(wdir + 'hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p', 'wb'))
    dst = src + "_acc_" + str(np.max(h.history['val_acc']))
    os.rename(src + '.h5', dst + '.h5')
    # net.save('Final_' + dst + '.h5')
    # dump(h.history, open('hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    with open('1.2.3_trace_rotation_diff.txt', 'a') as file:
        file.seek(0, 2)
        file.write(src + '     ' + str(np.max(h.history['val_acc'])) + '\n')
    return net, h


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
# train_speck_distinguisher(num_train=2 ** 18, num_eval=2 ** 18, num_epochs=2, num_rounds=14, depth=10,
#                           batch_size=2 ** 14, pairs=2, num_block=8, k=1, diff=(0x0, 0x0002))

# 单比特
# for rotation in range(15):
#     rotation = rotation + 1
#     for i in range(16):
#         diff = (1 << i)
        # print(hex(diff))
       # train_speck_distinguisher(num_train=2 ** 23, num_eval=2 ** 18, num_epochs=10, num_rounds=13, depth=10,
        #                          batch_size=2 ** 15, pairs=1, num_block=8, k=rotation, diff=(0x0, diff))

# 双比特遍历
# demo = [10,11,12,13,14,15]
# for rotation in demo:
#     for i in range(15):
#         for j in range(15 - i):
#             diff = (1 << i) ^ (1 << (j + 1 + i))
#             print(hex(diff))
#             train_speck_distinguisher(num_train=2 ** 23, num_eval=2 ** 18, num_epochs=10, num_rounds=13, depth=10,
#                                       batch_size=2 ** 15, pairs=1, num_block=8, k=rotation, diff=(0x0, diff))


# 较好路径单比特差分[(1,0x0004),(15,0x0002),(5,0x0040),(11,0x0002),(6,0x0080),(10,0x0002)]，前两个为最高0.7
# 较好路径双比特差分[(1,0x0006),(15,0x0003),(15,8002),(1,0x0005),(6,0x0082),(10,0x0802),(5,0x0042),(11,0x1002),(4,0x0022),(12,0x2002)]，都在0.6左右
# (15,0x0003),(15,8002),(10,0x0802),(11,0x1002)(12,0x2002)]，
demo = [(1,0x0004)]
for i in demo:
    k = i[0]
    diff= i[1]
    train_speck_distinguisher(num_train=2 ** 24, num_eval=2 ** 18, num_epochs=60, num_rounds=16, depth=10,
                              batch_size=2 ** 16, pairs=36, num_block=2, k=k, diff=(0x0, diff))
