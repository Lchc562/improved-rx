import speck as sp
# import simeck
import simon
import numpy as np
from keras.models import model_from_json, load_model
from os import urandom
import copy
import os
import gc
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
block_size = 32


# type = 0, cal sen_0
# type = 1, cal sen_1
# type = 2, cal sen_01
def make_target_bit_diffusion_data(X, id=0, type=0):
    n = X.shape[0]
    # print(X)
    masks = np.frombuffer(urandom(n), dtype=np.uint8) & 0x1
    masked_X = copy.deepcopy(X)
    if type == 0:  # 密文对中的c0变化
        masked_X[:, block_size - 1 - id] = X[:, block_size - 1 - id] ^ masks
    elif type == 1:  # 密文对中的c1变化
        masked_X[:, block_size * 2 - 1 - id] = X[:, block_size * 2 - 1 - id] ^ masks
    else:  # 密文对中的c0c1一起变化，也就是差分不变
        masked_X[:, block_size - 1 - id] = X[:, block_size - 1 - id] ^ masks
        masked_X[:, block_size * 2 - 1 - id] = X[:, block_size * 2 - 1 - id] ^ masks
    # print("盲化后：",masks,type,'\n',masked_X)
    # input("pause")
    return masked_X


# X, Y = sp.make_train_data(n=2, nr=1, diff=(0x0, 0x4))
# print("原始数据：",X)
# make_target_bit_diffusion_data(X, id=0, type=0)


def test_bits_sensitivity(n=10 ** 7, nr=7, type=0, k=15, diff=(0x0040, 0x0), folder='./bits_sensitive_res/'):
    acc = np.zeros(block_size + 1)
    X, Y = simon.make_datas_demo(n=n, rounds=nr, k=k, diff=diff)
    # json_file = open('./simon_model/data14_diff1_15_03_r12_depth10_0.6270.h5', 'r')
    # json_model = json_file.read()
    # net = model_from_json(json_model)
    # net.load_weights(net_path)
    net = load_model('./simon_model/data05_diff1_15_03_r11_depth10_0.91.h5')
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])
    loss, acc[block_size] = net.evaluate(X, Y, batch_size=10000, verbose=0)
    print('The initial acc is ', acc[block_size])

    for i in range(16):
        masked_X = make_target_bit_diffusion_data(X, id=i+16, type=type)
        loss, acc[i] = net.evaluate(masked_X, Y, batch_size=10000, verbose=0)
        print('cur bit position is ', i)
        print('the decrease of the acc is ', acc[block_size] - acc[i])
        del masked_X
        gc.collect()

    np.save(folder + 'simon32_data05_15_03_' + str(nr) + 'r_type' + str(type) + '_bit_sensitivity.npy', acc)

# net_path = './saved_model/teacher/0x0040-0x0/5_distinguisher.h5'
# net_path = './saved_model/teacher/0x0040-0x0/6_distinguisher.h5'
net_path = './simon_model/data05_diff1_15_03_r11_depth10_0.91.h5'
# net_path = './saved_model/teacher/0x0040-0x0/net8_small.h5'
folder = 'bit_sensetive_result/'
for i in range(1):
    test_bits_sensitivity(n=10**0, nr=11, k=15, diff=(0x0, 0x3), folder=folder, type=2)
