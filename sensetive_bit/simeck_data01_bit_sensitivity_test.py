import gc
import simeck
import numpy as np
from keras.models import load_model
from os import urandom
import copy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
block_size = 32


# type = 0, cal sen_0
# type = 1, cal sen_1
# type = 2, cal sen_01
# adapt data 05_bits
def make_target_bit_diffusion_data(X, id=0, type=0):
    n = X.shape[0]
    masks = np.frombuffer(urandom(n), dtype=np.uint8) & 0x1
    masked_X = copy.deepcopy(X)
    if type == 0:  # 密文对中的c0变化
        masked_X[:, block_size - 1 - id] = X[:, block_size - 1 - id] ^ masks
    elif type == 1:  # 密文对中的c1变化
        masked_X[:, block_size * 2 - 1 - id] = X[:, block_size * 2 - 1 - id] ^ masks
    else:  # 密文对中的c0c1一起变化，也就是差分不变
        masked_X[:, block_size - 1 - id] = X[:, block_size - 1 - id] ^ masks
        masked_X[:, block_size * 2 - 1 - id] = X[:, block_size * 2 - 1 - id] ^ masks
    return masked_X


def fun(X, id=0, type=0):
    cipher_01, cipher_02, cipher_11, cipher_12 = X[0], X[1], X[2], X[3]
    n = cipher_01.shape[0]
    mask = np.frombuffer(urandom(n), dtype=np.uint8) & 0x1
    masks = mask.astype(np.uint32) << id
    # print("n,id,type",n,id,type)
    # print("初始密文：",cipher_01,cipher_02)
    print('mask盲化值:',masks,id)
    print(cipher_01)
    if type == 0:  # 密文对中的c0变化
        if id <= 15:
            cipher_02 = cipher_02 ^ masks
        else:
            cipher_01 = cipher_01 ^ masks
    elif type == 1:  # 密文对中的c1变化
        if id <= 15:
            cipher_12 = cipher_12 ^ masks
        else:
            cipher_11 = cipher_11 ^ masks
    else:  # 密文对中的c0c1一起变化，也就是差分不变
        if id <= 15:
            cipher_02 = cipher_02 ^ masks
            cipher_12 = cipher_12 ^ masks
        else:
            cipher_01 = cipher_01 ^ masks
            cipher_11 = cipher_11 ^ masks
    print(cipher_01)
    # input('pause')
    ct_diff_0111 = cipher_01 ^ cipher_11
    ct_diff_0212 = cipher_02 ^ cipher_12
    # ctr_diff1 = simon.make_diff_decry([cipher_01, cipher_02, cipher_11, cipher_12], 1)
    # ctr_diff2 = simon.make_diff_decry([cipher_01, cipher_02, cipher_11, cipher_12], 2)
    ctr_diff1 = simeck.make_diff_decry([cipher_01, cipher_02, cipher_11, cipher_12], 1)
    ctr_diff2 = simeck.make_diff_decry([cipher_01, cipher_02, cipher_11, cipher_12], 2)

    # data 14
    X = simeck.RXDR_convert_to_bits([ct_diff_0111, ct_diff_0212, cipher_01, cipher_02,
                                    cipher_11, cipher_12, ctr_diff1, ctr_diff2], 8)
    return X


# X, Y = sp.make_train_data(n=2, nr=1, diff=(0x0, 0x4))
# print("原始数据：",X)
# make_target_bit_diffusion_data(X, id=0, type=0)

def test_bits_sensitivity(n=10 ** 7, nr=7, type=0, k=15, diff=(0x0040, 0x0),folder='./bits_sensitive_res/'):
    acc = np.zeros(block_size + 1)
    X, Y, demo = simeck.sensetive_make_train_data(n=n, nr=nr, k=k, diff=diff)

    net = load_model('./simeck_model/simeck32_data14_1_4_r13_depth10_acc_0.6990966796875.h5')
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    loss, acc[block_size] = net.evaluate(X, Y, batch_size=10000, verbose=0)
    print('The initial acc is ', acc[block_size])

    for i in range(block_size):
        # masked_X = make_target_bit_diffusion_data(demo, id=i, type=type)
        masked_X = fun(demo, id=i, type=type)
        loss, acc[i] = net.evaluate(masked_X, Y, batch_size=10000, verbose=0)
        print('cur bit position is ', i)
        print('the decrease of the acc is ', acc[block_size] - acc[i])
        del masked_X
        gc.collect()

    np.save(folder + 'simeck32_data14_1_4_' + str(nr) + 'r_type' + str(type) + '_bit_sensitivity.npy', acc)

folder = 'bit_sensetive_result/'
for i in range(3):
    test_bits_sensitivity(n=10**6, nr=13, k=1, diff=(0x0, 0x4), folder=folder, type=i)

