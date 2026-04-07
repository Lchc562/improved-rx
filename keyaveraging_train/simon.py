#!/usr/bin/python3
import gc

import numpy as np
from os import urandom
from collections import deque
from math import log2

inDIff = []
inDIff.append((0, 0x0040))
inDIff.append([])
inDIff.append((0x0100, 0x0440))
inDIff.append((0x0440, 0x1000))  # 3
inDIff.append((0x1000, 0x4440))  # 4
inDIff.append((0x4000, 0x5101))  # 5


def WORD_SIZE():
    return 16


def ALPHA():
    return (1);


def BETA():
    return (8);


def GAMMA():
    return (2);


z_0 = 0b01100111000011010100100010111110110011100001101010010001011111
MOD_MASK = (2 ** WORD_SIZE()) - 1

c = MOD_MASK ^ 3


def rol(x, k):
    return (((x << k) & MOD_MASK) | (x >> (WORD_SIZE() - k)));  # 需要&上mask,猜测是因为x的比特数量，如果不是word_size个则可以修正


def ror(x, k):
    return ((x >> k) | ((x << (WORD_SIZE() - k)) & MOD_MASK));


def text_rol(x, k, bit_num=32):
    MOD_MASK1 = (2 ** bit_num) - 1
    return (((x << k) & MOD_MASK1) | (x >> (bit_num - k)));  # 需要&上mask,猜测是因为x的比特数量，如果不是word_size个则可以修正


def text_ror(x, k, bit_num=32):
    MOD_MASK1 = (2 ** bit_num) - 1
    return ((x >> k) | ((x << (bit_num - k)) & MOD_MASK1));


def expand_keys(key, rounds):
    k_tmp = deque(key)  # 创建一个双向队列
    keys = []
    for i in range(rounds):
        rs_3 = ((k_tmp[0] << (WORD_SIZE() - 3)) + (k_tmp[0] >> 3)) & MOD_MASK
        # m= 4
        rs_3 = rs_3 ^ k_tmp[2]
        #
        rs_1 = ((rs_3 << (WORD_SIZE() - 1)) + (rs_3 >> 1)) & MOD_MASK
        tmp = rs_3 ^ rs_1
        z = (z_0 >> (i % 62)) & 1
        new_k = z ^ c ^ k_tmp[3] ^ tmp
        # print(i,'   ',hex(z^c))
        keys.append(k_tmp.pop())
        k_tmp.appendleft(new_k)
    return keys


def round_function(x):
    ls_1_x = ((x >> (WORD_SIZE() - 1)) + (x << 1)) & MOD_MASK
    ls_2_x = ((x >> (WORD_SIZE() - 2)) + (x << 2)) & MOD_MASK
    ls_8_x = ((x >> (WORD_SIZE() - 8)) + (x << 8)) & MOD_MASK
    # print(ls_1_x,ls_2_x,ls_8_x)
    f_value = (ls_1_x & ls_8_x) ^ ls_2_x
    return f_value


def one_round_encrypt(plain, key):
    x, y = plain[0], plain[1]

    f_value = round_function(x)
    xor_1 = f_value ^ y
    y = x
    x = xor_1 ^ key
    return x, y


def encryption(plain_pair, keys):
    x, y = plain_pair[0], plain_pair[1]
    for key in keys:
        x, y = one_round_encrypt((x, y), key)
    return x, y


# def one_round_decrypt(cipher, key):
#     x, y = cipher[0], cipher[1]
#     f_value = round_function(y)
#     xor_1 = f_value ^ x
#     x = y
#     y = xor_1 ^ key
#     return x, y

def one_round_decrypt(cipher, key):
    # print("密文与密钥：",cipher,key)
    x, y = cipher[0], cipher[1]
    f_value = round_function(y)
    xor_1 = f_value ^ x
    # print("f_value,xor_1:",f_value,xor_1)
    x = y  # 这里没有广播机制，所以x的数量只有1，和后面的异或得到的y数量不同，问题就出在这里
    # print("jiemiinter", xor_1, key)
    y = xor_1[:, np.newaxis] ^ key
    # print("test",x,y)
    x = np.repeat(x, len(key))
    # print(xor_1, key)
    # print("inter", x)
    # print(y)
    # print(y.flatten(), "over\n")
    return x, y.flatten()


def decryption(cipher_pair, keys):
    x, y = cipher_pair[0], cipher_pair[1]
    for key in reversed(keys):
        x, y = one_round_decrypt((x, y), key)
    return x, y


def dec_one_round_eq(c, k):
    c0, c1 = c[0], c[1];
    t1 = c1 ^ k
    t0 = (rol(t1, ALPHA()) & rol(t1, BETA())) ^ rol(c1, GAMMA());
    c0 = c0 ^ t0;
    return (c1, c0);


def convert_to_bits(arr):
    X = np.zeros((4 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X


def RXDR_convert_to_bits(arr, num_word=4):
    X = np.zeros((num_word * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(num_word * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X


def diff_convert_to_bits(arr):
    X = np.zeros((2 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(2 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X


def value_diff_convert_to_bits(arr):
    X = np.zeros((3 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(3 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X


def make_datas_diff(n, rounds, diff=(0, 0x0040)):
    print("Make_data Input diff: ", diff)
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_random_samples = np.sum(Y == 0)
    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys = expand_keys(key, rounds)
    plain_01 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_02 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_11 = plain_01 ^ diff[0]
    plain_12 = plain_02 ^ diff[1]
    plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    cipher_01, cipher_02 = encryption((plain_01, plain_02), keys)
    cipher_11, cipher_12 = encryption((plain_11, plain_12), keys)
    delta_x = cipher_01 ^ cipher_11
    delta_y = cipher_02 ^ cipher_12
    X = diff_convert_to_bits([delta_x, delta_y])
    return X, Y


def check_vector():
    x, y = np.array([0x3cbd, 0x49a0], dtype=np.uint16), np.array([0xad88, 0x7563], dtype=np.uint16)
    key = np.array([[0xb4a1], [0xb9d5], [0xf340], [0x84cd]], dtype=np.uint16)

    keys = expand_keys(key, rounds=5)
    cipher_x1, cipher_x2 = encryption(x, keys)
    # hex_values = [[hex(element) for element in row] for row in [cipher_x1, cipher_x2]]
    # print(hex_values)
    # hex_values = [hex(element) for element in y]
    # print(hex_values)
    if ((y[0] == cipher_x1) & (y[1] == cipher_x2)):
        print("check_vector is no proplem")
    else:
        print("encryption error")


def make_datas(n, rounds, diff=(0, 0x0040)):
    print("Make_data Input diff: ", diff)
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_random_samples = np.sum(Y == 0)
    # key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    # keys = expand_keys(key, rounds)
    # 密钥方案，主密钥0x0123456789123456  0123为最高位，reshape(4,-1)后为[[0123],[4567],[8912],[3456]] [3456]作为了第一轮的密钥输出
    # 按照32/64 m=4的密钥方案  主密钥分为了k0~3   最先pop的是k0，耶就是最高位[3456],这在循环移位中非常重要

    key = np.frombuffer(urandom(8 * n), dtype=np.uint64)
    # print("%x"%key[0])
    key = key.view(np.uint16).reshape(4, -1)
    keys = expand_keys(key, rounds)

    # hex_values = [[hex(element) for element in tmp] for tmp in key]
    # print(hex_values)
    # hex_values = [[hex(element) for element in tmp] for tmp in keys]
    # print(hex_values)

    plain_01 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_02 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_11 = plain_01 ^ diff[0]
    plain_12 = plain_02 ^ diff[1]
    plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    cipher_01, cipher_02 = encryption((plain_01, plain_02), keys)
    cipher_11, cipher_12 = encryption((plain_11, plain_12), keys)
    X = convert_to_bits([cipher_01, cipher_02, cipher_11, cipher_12])
    return X, Y


def make_diff_decry(cts, rounds=1, cycle=None):
    if cycle is None:
        cycle = [1, 8, 2]
    ct0 = rol(cts[1], cycle[0]) & rol(cts[1], cycle[1]) ^ rol(cts[1], cycle[2]) ^ cts[0]
    ct1 = rol(cts[3], cycle[0]) & rol(cts[3], cycle[1]) ^ rol(cts[3], cycle[2]) ^ cts[2]
    if rounds == 1:
        return ct0 ^ ct1
    if rounds == 2:
        ct0 = rol(ct0, cycle[0]) & rol(ct0, cycle[1]) ^ rol(ct0, cycle[2]) ^ cts[1]
        ct1 = rol(ct1, cycle[0]) & rol(ct1, cycle[1]) ^ rol(ct1, cycle[2]) ^ cts[3]
        return ct0 ^ ct1


def data14_generate(cts, k=15, if_rotation=1, if_bit=1):
    cipher_01, cipher_02, cipher_11, cipher_12 = cts[0], cts[1], cts[2], cts[3]
    if if_rotation:
        cipher_01, cipher_02 = rol(cipher_01, k), rol(cipher_02, k)
    # 生成部分解密1/2轮数据
    ct_diff_0111 = cipher_01 ^ cipher_11
    ct_diff_0212 = cipher_02 ^ cipher_12
    # print(ct_diff_0111,ct_diff_0212)
    ctr_diff1 = make_diff_decry([cipher_01, cipher_02, cipher_11, cipher_12], 1)
    ctr_diff2 = make_diff_decry([cipher_01, cipher_02, cipher_11, cipher_12], 2)
    if if_bit:
        X = RXDR_convert_to_bits([ct_diff_0111, ct_diff_0212, cipher_01, cipher_02,
                                  cipher_11, cipher_12, ctr_diff1, ctr_diff2], 8)
    else:
        X = [ct_diff_0111, ct_diff_0212, cipher_01, rol(cipher_02, k),
             cipher_11, cipher_12, ctr_diff1, ctr_diff2]
    return X


# Test_RXDR_Data_Demo
def make_datas_demo(n, rounds, diff=(0, 0x0006), k=1):
    # print("Make_data Input diff: ", diff)
    check_vector()
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_random_samples = np.sum(Y == 0)
    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    key1 = rol(key, k)
    key1[3] = key1[3] ^ diff[1]
    keys = expand_keys(key, rounds)
    keys1 = expand_keys(key1, rounds)
    keys_f1 = expand_keys(key, rounds - 1)
    keys1_f1 = expand_keys(key1, rounds - 1)
    # print(key)
    # print(keys)
    # print(keys1)
    # for i in range(len(keys)):
    #     # print(keys[i],keys1[i])
    #     print('密钥差分   ',i,'    ',rol(keys[i],k) ^ keys1[i], hex((rol(keys[i],k) ^ keys1[i])[0]))

    plain_01 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_02 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_11 = rol(plain_01, k) ^ diff[0]
    plain_12 = rol(plain_02, k) ^ diff[1]
    plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)

    cipher_01, cipher_02 = encryption((plain_01, plain_02), keys)
    cipher_11, cipher_12 = encryption((plain_11, plain_12), keys1)
    cipher_01_f1, cipher_02_f1 = encryption((plain_01, plain_02), keys_f1)
    cipher_11_f1, cipher_12_f1 = encryption((plain_11, plain_12), keys1_f1)

    del plain_01, plain_02, plain_11, plain_12
    gc.collect()
    # ct_diff_0121 = cipher_01 ^ cipher_21
    # ct_diff_0222 = cipher_02 ^ cipher_22
    # ct_diff_1131 = rol(cipher_11 ^ cipher_31,k)
    # ct_diff_1232 = rol(cipher_12 ^ cipher_32,k)

    # X = RXDR_convert_to_bits([rol(cipher_01, k), rol(cipher_02, k), cipher_11, cipher_12], 4)

    # train_keyaveraging
    # X = [rol(cipher_01, k), rol(cipher_02, k), cipher_11, cipher_12]
    X = [cipher_01, cipher_02, cipher_11, cipher_12]
    X_f1 = [rol(cipher_01_f1, k), rol(cipher_02_f1, k), cipher_11_f1, cipher_12_f1]
    return X, X_f1, Y


# X, x_f1, Y = make_datas_demo(2, 20, k=15, diff=(0x0, 0x3))


# print(X,Y)
# print(len(X), len(X[0]))


def make_data_value_diff(n, rounds, diff=(0, 0x0040)):
    print("Make_data Input diff: ", diff)
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_random_samples = np.sum(Y == 0)
    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys = expand_keys(key, rounds)
    plain_01 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_02 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_11 = plain_01 ^ diff[0]
    plain_12 = plain_02 ^ diff[1]
    plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    cipher_01, cipher_02 = encryption((plain_01, plain_02), keys)
    cipher_11, cipher_12 = encryption((plain_11, plain_12), keys)
    delta_y = cipher_02 ^ cipher_12  # 与上面make_data不同的地方
    X = value_diff_convert_to_bits([cipher_01, cipher_11, delta_y])
    return X, Y
