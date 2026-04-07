# implements KeyAveraging and the creation of high-quality training data using the output of KeyAveraging.
import gc

import numpy as np
from keras.models import load_model
from pickle import dump

from net_multi_2019 import make_resnet, cyclic_lr, make_checkpoint
from os import urandom
import simon  # 直接调用运行了simon.py，里面直接运行的语句会被执行
# from pympler import asizeof
from pickle import dump

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model
# from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
import os
import tensorflow as tf

wdir = './keyaveraging_result/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
WORD_SIZE = simon.WORD_SIZE()


def key_average(ct0a, ct1a, ct0b, ct1b, keys, net, k=15, diff=(0x0, 0x3)):
    # 每次只进来一对明文
    keys1 = simon.rol(keys, k) ^ 0x3126
    print("相关密钥差分关系：", keys, keys1, simon.rol(keys, k) ^ keys1)
    print("one_round_decrypt start")
    pt0a, pt1a = simon.one_round_decrypt((ct0a, ct1a), keys)  # 这里出了问题，错误使用广播机制？？原本是一个明文所使用的所有秘钥解密一遍
    pt0b, pt1b = simon.one_round_decrypt((ct0b, ct1b), keys1)
    print("解密后结果:", [pt0a, pt1a, pt0b, pt1b], pt0a.shape)
    pt0a, pt1a = simon.rol(pt0a, k), simon.rol(pt1a, k)
    print("one_round_decrypt over")
    # print("用所有密钥解密一轮后",pt1a)
    # print([pt0a, pt1a, pt0b, pt1b])

    X = simon.convert_to_bits([pt0a, pt1a, pt0b, pt1b])
    # print("转换为比特前的大小(GB)：", asizeof.asizeof(X) / (2 ** 30))
    # print("X.shape_bit",X.shape)

    print("predict start")
    with strategy.scope():
        Z = net.predict(X, batch_size=2 ** 18).flatten()
        # tf.keras.backend.clear_session()
        print("predict over")
        Z = (Z / (1 - Z)).reshape(len(ct0a), -1)
        # print("计算值1", Z)
        v = np.average(Z, axis=1)
        del Z
        del X
        gc.collect()
        # print("计算值1", v)
        v = v / (v + 1)
        # Z_tensor = tf.convert_to_tensor(X)
    return v


# "with_teacher"表示使用已知的目标值（通常由教师或专家提供）来训练模型。这意味着
# 在生成训练集时，使用了与预期输出相关联的标签或目标值。在这种情况下，生成的训练集包
# 括输入数据和与之对应的目标输出。这里的teacher指的是key_average
def make_trainset_with_teacher(n, net, nr=12, k=15, diff=(0x0, 0x3), num_keys=1000, keys=None, counts=16):
    change_keys = (keys is None)
    X, X_f1, Y = simon.make_datas_demo(n, nr, k=k, diff=diff)  # 最外层不是np类型数据，用np函数时会出现奇怪的分割错误，不能保证完整性
    # X = [[29456, 39614],
    #      [45972, 21667],
    #      [8257, 23961],
    #      [17805, 31575]]
    # X = [[29456],
    #      [45972],
    #      [8257],
    #      [17805]]
    X = np.array(X, dtype=np.uint16)  # 必要操作，保证完整性不出现错误
    X_f1 = np.array(X_f1, dtype=np.uint16)
    # print("原密文：", X,X_f1)
    X = np.array_split(X, counts, axis=1)
    # print("分割后密文：", X)
    allkeys = np.arange(0, 2 ** WORD_SIZE, dtype=np.uint16)
    # replace=False保证选取不重复，但是这里直接用全秘钥空间不就好了
    if change_keys: keys = np.random.choice(allkeys, num_keys, replace=False)
    # keys = np.array([1688,32584,5446])
    # print("密钥选择：", keys)
    print("start genarate teacher_data")
    Z = []
    for i in range(counts):
        print("迭代进度", i + 1, "/", counts)
        Z = np.append(Z, key_average(X[i][0], X[i][1], X[i][2], X[i][3], allkeys, net, k=k, diff=diff))
        # print(Z)
    print(X, X_f1)
    X = simon.convert_to_bits(np.concatenate(X, axis=1))  # X进行了分割，维度变成了三维,拼接之后降了一个维度
    X_f1 = simon.convert_to_bits(X_f1)  # 只有两个维度
    with strategy.scope():
        Y_f1 = net.predict(X_f1, batch_size=2 ** 18).flatten()
    print("genarate over")
    return X, X_f1, Y, Y_f1, Z


# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
# with strategy.scope():
#     model_teacher = load_model('best11k15diff_0_3.h5', compile=True)
#     x, y, z = make_trainset_with_teacher(2**16, model_teacher, k=15, diff=(0x0, 0x3), nr=10,
#                                      num_keys=2**10, keys=None)
# print(x)
# print("TEACHER", z)


# print(len(x))

def train_speck_distinguisher(num_train, num_eval, num_epochs, num_keys=100, num_rounds=7, depth=1, batch_size=10 ** 5,
                              num_block=4, k=1, diff=(0x0, 0x6), counts1=16, counts2=1):
    with strategy.scope():
        net = make_resnet(num_blocks=num_block, depth=depth, reg_param=10 ** -5)
        # 编译模型：指定优化器，损失函数和评估指标
        net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    model_teacher = load_model('best11_k15diff_0_3.h5', compile=False)

    X, X_f1, Y, Y_f1, Z = make_trainset_with_teacher(num_train, model_teacher, k=k, diff=diff, nr=num_rounds,
                                                     num_keys=num_keys, keys=None, counts=counts1)
    np.savez('x_xf1_y_yf1_z_array', x=X, x_f1=X_f1, y=Y, y_f1=Y_f1, z=Z)
    X_eval, X_eval_f1, Y_eval, Y_f1_eval, Z_eval = make_trainset_with_teacher(num_eval, model_teacher, k=k, diff=diff,
                                                                              nr=num_rounds, num_keys=num_keys,
                                                                              keys=None, counts=counts2)
    np.savez('x_xf1_y_yf1_z_eval_array', x=X_eval, x_f1=X_eval_f1, y=Y_eval, y_f1=Y_f1_eval, z=Z_eval)
    Z, Z_eval = np.round(Z).astype(int), np.round(Z_eval).astype(int)
    src = wdir + 'simon_keyav_' + str(simon.WORD_SIZE() * 2) + '_data15_diff1_15_03' + '_r' + str(
        num_rounds) + '_depth' + str(depth)
    check = make_checkpoint(src + '.h5')
    # check = make_checkpoint('data15_diff1_15_50090003_r6_depth10_' + '.h5')

    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
    h = net.fit(X, Z, epochs=num_epochs, batch_size=batch_size, validation_data=(X_eval, Z_eval), callbacks=[lr, check],
                shuffle=True)
    dump(h.history, open(wdir + 'hist' + str(num_rounds) + 'r_depth' + str(depth) + '.p', 'wb'))
    print("Best validation accuracy: ", np.max(h.history['val_acc']))
    dst = src + "_acc_" + str(np.max(h.history['val_acc']))
    os.rename(src + '.h5', dst + '.h5')
    return net, h


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

# 步骤2: 创建MirroredStrategy，使用多GPU
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
print("test_kernel_position")
# train_speck_distinguisher(num_train=2 ** 9, num_eval=2 ** 9, batch_size=2 ** 15, num_keys=2**16, num_epochs=60,
#                           num_rounds=12, depth=1, num_block=4, k=15, diff=(0x0, 0x0003), counts1=2**0, counts2=2**0)

train_speck_distinguisher(num_train=2 ** 16, num_eval=2 ** 14, batch_size=2 ** 15, num_keys=2 ** 16, num_epochs=60,
                          num_rounds=12, depth=1, num_block=4, k=15, diff=(0x0, 0x0003), counts1=2 ** 7, counts2=2 ** 3)
