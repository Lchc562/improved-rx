import simon  
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
bs = 16384
wdir = './basic_result/'
# wdir = './rotation/'

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
    model.summary()

    return (model)


# make residual tower of convolutional blocks
def make_resnet(num_blocks=4, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3, depth=10,
                reg_param=0.0001, final_activation='sigmoid'):
    # Input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size,))
    rs = Reshape((num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)
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
        shortcut = Add()([shortcut, conv2])  
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
    model.summary()
    return model


def train_speck_distinguisher(num_train, num_eval, num_epochs, pairs=2, num_rounds=7, depth=1, batch_size=10 ** 5, num_block=4,
                              k=1, diff=(0x0, 0x6)):
    pairs = pairs
    # create the network
    # print("1")
    print(k, diff)
    with strategy.scope():
        net = make_resnet1(num_blocks=num_block, pairs=pairs, depth=depth, reg_param=10 ** -5)
        # 编译模型：指定优化器，损失函数和评估指标
        custom_op = tf.keras.optimizers.Adam(learning_rate=0.00001)
        net.compile(optimizer=custom_op, loss='mse', metrics=['acc'])
        # net.compile(optimizer='adam', loss='mse', metrics=['acc'])

    # X, Y = simon.make_datas_demo(num_train, num_rounds, k=k, diff=diff)
    # X_eval, Y_eval = simon.make_datas_demo(num_eval, num_rounds, k=k, diff=diff)
    X, Y = simon.make_datas_pairs(num_train, num_rounds, pairs=pairs, k=k, diff=diff)
    X_eval, Y_eval = simon.make_datas_pairs(num_eval, num_rounds, pairs=pairs, k=k, diff=diff)
    src = wdir + 'simon' + str(simon.WORD_SIZE() * 2) + '_data27_pairs36_' + str(k) + '_' + str(hex(diff[1])[2:]) + '_r' + str(
          num_rounds) + '_depth' + str(depth)
    check = make_checkpoint(src + '.h5')
    # check = make_checkpoint('data15_diff1_15_50090003_r6_depth10_' + '.h5')

    # lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    h = net.fit(X, Y, epochs=num_epochs, batch_size=batch_size, validation_data=(X_eval, Y_eval),
                callbacks=[check], shuffle=True)
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


# 步骤1：查看GPU设备情况
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
train_speck_distinguisher(num_train=2 ** 10, num_eval=2 ** 10, num_epochs=20, num_rounds=1, depth=10,
                          pairs=28, batch_size=2 ** 18, num_block=3, k=12, diff=(0x0, 0x2002))
# train_speck_distinguisher(num_train=2 ** 24, num_eval=2 ** 18, num_epochs=20, num_rounds=15, depth=10,
#                           pairs=36, batch_size=2 ** 16, num_block=2, k=4, diff=(0x0, 0x0022))

# (1,0x6)(15,0x3)(3,0x0102)(13,0x4020)(4,0x0022)(12,0x2002)
# demo = [(1,0x6),(15,0x3),(3,0x0102),(13,0x4020)]
# # demo = [(15,0x3)]
# for j in demo:
#     k = j[0]
#     diff_r = j[1]
#     train_speck_distinguisher(num_train=2 ** 24, num_eval=2 ** 18, num_epochs=40,num_rounds=13, depth=10,
#                               pairs=28, batch_size=2 ** 16, num_block=3, k=k, diff=(0x0, diff_r))
