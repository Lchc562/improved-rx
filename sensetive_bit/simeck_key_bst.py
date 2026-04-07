import gc
import simeck
import numpy as np
from keras.models import load_model
from os import urandom
import copy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
block_size = 16


# type = 0, cal sen_0
# type = 1, cal sen_1
# type = 2, cal sen_01
# adapt data 05_bits

def fun(X, key0, key1, id=0,type=0):
    cipher_01, cipher_02, cipher_11, cipher_12 = X[0], X[1], X[2], X[3]
    n = cipher_01.shape[0]
    # 掩码随机 或者都是1进行改变
    if type==0:
        mask = np.frombuffer(urandom(2 * n), dtype=np.uint16) & 0x1
    elif type==1:
        mask = np.ones(n, dtype=np.uint32)
    masks = mask.astype(np.uint32) << id
    # print("n,id,type",n,id,type)
    # print("初始密文：",cipher_01,cipher_02)
    print('mask盲化值:', masks, id)
    # print(cipher_01)
    # 改变密钥，使对应位置比特错误
    print("掩码前密钥",key0,key1)
    key0 = key0 ^ masks
    key1 = key1 ^ masks
    print("掩码后密钥",key0,key1)
    # 使用错误密钥进行解密
    cipher_01, cipher_02 = simeck.dec_one_round_simeck((cipher_01, cipher_02), key0)
    cipher_11, cipher_12 = simeck.dec_one_round_simeck((cipher_11, cipher_12), key1)


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

def test_bits_sensitivity(n=10 ** 7, nr=7, k=1, diff=(0x0, 0x0004), folder='./bits_sensitive_res/',type=0):
    acc = np.zeros(block_size + 1)
    X, Y, demo, key0, key1 = simeck.make_data_keybst(n=n, nr=nr - 1, k=k, diff=diff)
    net = load_model('./simeck_model/simeck32_data14_1_4_r15_depth10_acc_0.5118331909179688.h5')
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    loss, acc[block_size] = net.evaluate(X, Y, batch_size=10000, verbose=0)
    print('The initial acc is ', acc[block_size])

    del X
    # 得到nr轮密文, 更新demo
    demo[0], demo[1] = simeck.encrypt_simeck((demo[0], demo[1]), [key0])
    demo[2], demo[3] = simeck.encrypt_simeck((demo[2], demo[3]), [key1])

    for i in range(block_size):
        # masked_X = make_target_bit_diffusion_data(demo, id=i, type=type)
        masked_X = fun(demo, key0, key1, id=i,type=type)
        loss, acc[i] = net.evaluate(masked_X, Y, batch_size=10000, verbose=0)
        print('cur bit position is ', i)
        print('the decrease of the acc is ', acc[block_size] - acc[i])
        del masked_X
        gc.collect()

    np.save(folder + 'simeck32_data14_1_4_' + str(nr) + 'r_type' + str(type) + '_bit_sensitivity.npy', acc)


folder = 'bit_sensetive_result/'
for i in range(2):
    test_bits_sensitivity(n=10 ** 6, nr=16, k=1, diff=(0x0, 0x4), folder=folder,type=i)
