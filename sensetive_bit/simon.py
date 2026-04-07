#!/usr/bin/python3
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
        keys.append(k_tmp.pop())
        k_tmp.appendleft(new_k)
    return keys


def round_function(x):
    ls_1_x = ((x >> (WORD_SIZE() - 1)) + (x << 1)) & MOD_MASK
    ls_2_x = ((x >> (WORD_SIZE() - 2)) + (x << 2)) & MOD_MASK
    ls_8_x = ((x >> (WORD_SIZE() - 8)) + (x << 8)) & MOD_MASK
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


def one_round_decrypt(cipher, key):
    x, y = cipher[0], cipher[1]
    f_value = round_function(y)
    xor_1 = f_value ^ x
    x = y
    y = xor_1 ^ key
    return x, y


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


# make_datas(2,5)


# def make_RXDR_data(n, rounds, k=1, diff=(0x0000, 0x0006)):
#     """
#     :param k: 旋转位数
#     :return:
#     """
#     check_vector()
#     Y = np.frombuffer(urandom(n), dtype=np.uint8)
#     Y = Y & 1
#     num_random_samples = np.sum(Y == 0)
#
#     key0_tmp = np.frombuffer(urandom(8 * n), dtype=np.uint64)
#     # print("%x"%key0_tmp[0],"%x"%key0_tmp[1])
#     key1_tmp = text_rol(key0_tmp, k,64) ^ diff[1]
#     # key1_tmp = text_rol(key0_tmp, k, 64)
#     # print("%x"%key1_tmp[0],"%x"%key1_tmp[1])
#     key0_tmp = key0_tmp.view(np.uint16)
#     key1_tmp = key1_tmp.view(np.uint16)
#
#     key0 = np.zeros((4, n), dtype=np.uint16)
#     key1 = np.zeros((4, n), dtype=np.uint16)
#     # 遍历 input 中的元素，根据 i%4 的值放入不同的 result 行
#     for i in range(4 * n):
#         row = i % 4# 计算要放入的 result 行
#         col = i // 4  # 计算要放入的 result 列
#         key0[row][col] = key0_tmp[i]
#     del key0_tmp
#     for i in range(4 * n):
#         row = i % 4
#         col = i // 4  # 计算要放入的 result 列
#         key1[row][col] = key1_tmp[i]
#     del key1_tmp
#
#     keys0 = expand_keys(key0, rounds)
#     keys1 = expand_keys(key1, rounds)
#
#     plain_0 = np.frombuffer(urandom(4 * n), dtype=np.uint32)  # 生成32bits明文，例如：0xdc1e81bd 81bd为低16位
#     plain_1 = text_rol(plain_0,1,32) ^ ((diff[0]<<16) | diff[1]) # 先不考虑差分，0xdc1e81bd <<< 1 = 0xb83d037b
#
#     plain_02 = plain_0.view(np.uint16)[::2]  # 不考虑差分view把uint32分为了2个uint16，把低16位放在了偶数位置，所以按照明文左右区分的话，偶数位的16位应该是明文的右半部分
#     plain_01 = plain_0.view(np.uint16)[1::2]
#     plain_02 = plain_02.copy()
#     plain_01 = plain_01.copy()
#     del plain_0
#     # print(plain_01)
#     plain_01[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
#     plain_02[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
#
#     plain_12 = plain_1.view(np.uint16)[::2]
#     plain_11 = plain_1.view(np.uint16)[1::2]
#     plain_12 = plain_12.copy()
#     plain_11 = plain_11.copy()
#     del plain_1
#     # print(plain_11)
#     plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
#     plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
#     # hex_values = [hex(element) for element in plain_0]
#     # print(hex_values)
#
#     plain_2 = np.frombuffer(urandom(4 * n), dtype=np.uint32)
#     plain_3 = text_rol(plain_2,1,32) ^ ((diff[0]<<16) | diff[1])
#
#     plain_22 = plain_2.view(np.uint16)[::2]
#     plain_21 = plain_2.view(np.uint16)[1::2]
#     plain_22 = plain_22.copy()
#     plain_21 = plain_21.copy()
#     del plain_2
#     # print(plain_21)
#     plain_21[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
#     plain_22[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
#
#     plain_32 = plain_3.view(np.uint16)[::2]
#     plain_31 = plain_3.view(np.uint16)[1::2]
#     plain_32 = plain_32.copy()
#     plain_31 = plain_31.copy()
#     del plain_3
#     # print(plain_21)
#     plain_31[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
#     plain_32[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
#
#     # 加密，转换
#     cipher_01, cipher_02 = encryption((plain_01, plain_02), keys0)
#     cipher_11, cipher_12 = encryption((plain_11, plain_12), keys1)
#     cipher_21, cipher_22 = encryption((plain_21, plain_22), keys0)
#     cipher_31, cipher_32 = encryption((plain_31, plain_32), keys1)
#     # print(cipher_21)
#     # print([cipher_01, cipher_02, cipher_11, cipher_12, cipher_21, cipher_22, cipher_31, cipher_32])
#     X = RXDR_convert_to_bits([cipher_01, cipher_02, cipher_11, cipher_12, cipher_21, cipher_22, cipher_31, cipher_32])
#     return X, Y

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


def sensetive_make_datas_demo(n, rounds, k=15, diff=(0x0, 0x3), id=0, type=None):
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

    plain_01 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_02 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_11 = rol(plain_01, k) ^ diff[0]
    plain_12 = rol(plain_02, k) ^ diff[1]
    plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)

    cipher_01, cipher_02 = encryption((plain_01, plain_02), keys)
    cipher_11, cipher_12 = encryption((plain_11, plain_12), keys1)
    del plain_01, plain_02, plain_11, plain_12

    cipher_01, cipher_02 = rol(cipher_01, k), rol(cipher_02, k)
    demo = [cipher_01, cipher_02, cipher_11, cipher_12]
    # 生成部分解密1/2轮数据
    ct_diff_0111 = cipher_01 ^ cipher_11
    ct_diff_0212 = cipher_02 ^ cipher_12
    ctr_diff1 = make_diff_decry([cipher_01, cipher_02, cipher_11, cipher_12], 1)
    ctr_diff2 = make_diff_decry([cipher_01, cipher_02, cipher_11, cipher_12], 2)

    # data 14
    X = RXDR_convert_to_bits([ct_diff_0111, ct_diff_0212, cipher_01, cipher_02,
                              cipher_11, cipher_12, ctr_diff1, ctr_diff2], 8)
    return X, Y, demo


# X, Y = sensetive_make_datas_demo(3, 6, k=1, diff=(0x0, 0x6), id=0, type=None)
# print(X, Y)
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


# Test_RXDR_Data_Demo
def make_datas_demo(n, rounds, diff2=(0, 0x0006), k2=1, k=15, diff=(0x0, 0x3)):
    # print("Make_data Input diff: ", diff)
    check_vector()
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_random_samples = np.sum(Y == 0)
    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    key1 = rol(key, k)
    key1[3] = key1[3] ^ diff[1]
    key2 = rol(key, k2)
    key2[3] = key2[3] ^ diff2[1]

    keys = expand_keys(key, rounds)
    keys1 = expand_keys(key1, rounds)
    keys2 = expand_keys(key2, rounds)

    plain_01 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_02 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_11 = rol(plain_01, k) ^ diff[0]
    plain_12 = rol(plain_02, k) ^ diff[1]
    plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)

    # plain_21 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # plain_22 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # plain_31 = rol(plain_21, k) ^ diff[0]
    # plain_32 = rol(plain_22, k) ^ diff[1]
    # plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    # plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)

    # plain_41 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # plain_42 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # plain_51 = rol(plain_41, k2) ^ diff2[0]
    # plain_52 = rol(plain_42, k2) ^ diff2[1]
    # plain_51[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    # plain_52[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)

    # plain_61 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # plain_62 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # plain_71 = rol(plain_61, k2) ^ diff2[0]
    # plain_72 = rol(plain_62, k2) ^ 0x0001
    # plain_71[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    # plain_72[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    #
    # plain_81 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # plain_82 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    # plain_91 = rol(plain_81, k) ^ diff2[0]
    # plain_92 = rol(plain_82, k) ^ 0x8000
    # plain_91[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    # plain_92[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)

    cipher_01, cipher_02 = encryption((plain_01, plain_02), keys)
    cipher_11, cipher_12 = encryption((plain_11, plain_12), keys1)
    # cipher_21, cipher_22 = encryption((plain_21, plain_22), keys)
    # cipher_31, cipher_32 = encryption((plain_31, plain_32), keys1)
    # cipher_41, cipher_42 = encryption((plain_41, plain_52), keys)
    # cipher_51, cipher_52 = encryption((plain_51, plain_52), keys2)
    # cipher_61, cipher_62 = encryption((plain_61, plain_62), keys)
    # cipher_71, cipher_72 = encryption((plain_71, plain_72), keys2)
    # cipher_81, cipher_82 = encryption((plain_81, plain_82), keys)
    # cipher_91, cipher_92 = encryption((plain_91, plain_92), keys1)

    del plain_01, plain_02, plain_11, plain_12
    # del plain_41, plain_42, plain_51, plain_52

    # 0,1号旋转对移位为15,  4，5号旋转对移位为1  只能分别移动之后再链接
    # cipher_01, cipher_02 = rol(cipher_01, k2), rol(cipher_02, k2)
    # cipher_41, cipher_42 = rol(cipher_41, k2), rol(cipher_42, k2)
    # cipher_61, cipher_62 = rol(cipher_61, k2), rol(cipher_62, k2)
    # cipher_81, cipher_82 = rol(cipher_81, k2), rol(cipher_82, k)

    # cipher_01 = np.concatenate((rol(cipher_01, k), cipher_41))
    # cipher_02 = np.concatenate((rol(cipher_02, k), cipher_42))
    # cipher_11 = np.concatenate((cipher_11, cipher_51))
    # cipher_12 = np.concatenate((cipher_12, cipher_52))
    # Y = np.concatenate((Y,Y))

    # cipher_01 = np.concatenate((rol(cipher_01, k), cipher_41, cipher_61, cipher_81))
    # cipher_02 = np.concatenate((rol(cipher_02, k), cipher_42, cipher_62, cipher_82))
    # cipher_11 = np.concatenate((cipher_11, cipher_51, cipher_71, cipher_91))
    # cipher_12 = np.concatenate((cipher_12, cipher_52, cipher_72, cipher_92))
    # Y = np.concatenate((Y,Y,Y,Y))

    # ct_diff_0111 = rol(cipher_01, k) ^ cipher_11
    # ct_diff_0212 = rol(cipher_02, k) ^ cipher_12
    # ct_diff_2131 = rol(cipher_21, k) ^ cipher_31
    # ct_diff_2232 = rol(cipher_22, k) ^ cipher_32

    # ct_diff_0121 = cipher_01 ^ cipher_21
    # ct_diff_0222 = cipher_02 ^ cipher_22
    # ct_diff_1131 = rol(cipher_11 ^ cipher_31,k)
    # ct_diff_1232 = rol(cipher_12 ^ cipher_32,k)

    # ct_diff_01211131 = rol(cipher_01, k) ^ rol(cipher_21, k) ^ cipher_11 ^ cipher_31
    # ct_diff_02221232 = rol(cipher_02, k) ^ rol(cipher_22, k) ^ cipher_12 ^ cipher_32

    # 生成部分解密1/2轮数据
    # ct_diff_0111 = rol(cipher_01, k) ^ cipher_11
    # ct_diff_0212 = rol(cipher_02, k) ^ cipher_12
    # ctr_diff1 = make_diff_decry([rol(cipher_01, k), rol(cipher_02, k), cipher_11, cipher_12], 1)
    # ctr_diff2 = make_diff_decry([rol(cipher_01, k), rol(cipher_02, k), cipher_11, cipher_12], 2)

    # X = RXDR_convert_to_bits([ct_diff_0111, ct_diff_0212, ct_diff_2131, ct_diff_2232],4)
    # X = RXDR_convert_to_bits([cipher_01, cipher_02, cipher_21, cipher_22, cipher_11, cipher_12, cipher_31, cipher_32],
    #                          8)
    # X = RXDR_convert_to_bits([ct_diff_0121,ct_diff_0222,ct_diff_1131,ct_diff_1232],4)
    # X = RXDR_convert_to_bits([rol(cipher_01,k), rol(cipher_02,k), cipher_21, cipher_22, rol(cipher_11,k), rol(cipher_12,k), cipher_31, cipher_32],
    #                          8)
    # data 5
    X = RXDR_convert_to_bits([rol(cipher_01, k), rol(cipher_02, k), cipher_11, cipher_12], 4)
    # X = RXDR_convert_to_bits([rol(cipher_01,k), rol(cipher_02,k), rol(cipher_21,k), rol(cipher_22,k),
    #                           cipher_11, cipher_12, cipher_31, cipher_32,
    #                           ct_diff_0121,ct_diff_0222,ct_diff_1131,ct_diff_1232], 12)
    # X = RXDR_convert_to_bits([ct_diff_01211131, ct_diff_02221232], 2)

    # X = RXDR_convert_to_bits([cipher_01, cipher_02, cipher_11, cipher_12], 4)  # 多数据串联
    # X = RXDR_convert_to_bits([rol(cipher_01, k), rol(cipher_02, k), cipher_11, cipher_12,
    #                           cipher_41, cipher_42, cipher_51, cipher_52], 8)  # 多数据并联,4号数据已经旋转对齐
    # X = RXDR_convert_to_bits([rol(cipher_01, k), rol(cipher_02, k), cipher_11, cipher_12,
    #                           cipher_41, cipher_42, cipher_51, cipher_52,
    #                           cipher_61, cipher_62, cipher_71, cipher_72,
    #                           cipher_81, cipher_82, cipher_91, cipher_92,
    #                           ], 16)  # 多数据并联,4号数据已经旋转对齐

    # keyaveraging_train
    # X = [rol(cipher_01, k), rol(cipher_02, k), cipher_11, cipher_12]

    # add_decryption
    # data 14
    # X = RXDR_convert_to_bits([ct_diff_0111, ct_diff_0212, rol(cipher_01,k), rol(cipher_02,k),
    #                           cipher_11, cipher_12, ctr_diff1, ctr_diff2], 8)
    # data 15
    # X = RXDR_convert_to_bits([ct_diff_0111, ct_diff_0212, cipher_01, cipher_02,
    #                           cipher_11, cipher_12, ctr_diff1, ctr_diff2], 8)
    # data 16
    # X = RXDR_convert_to_bits([ct_diff_0111, ct_diff_0212, cipher_01, cipher_02,
    #                           cipher_11, cipher_12, ct_diff_0212, ctr_diff1,
    #                           ctr_diff1, ctr_diff2], 10)
    # data 17
    # X = RXDR_convert_to_bits([ct_diff_0111, ct_diff_0212, rol(cipher_01,k), rol(cipher_02,k)
    #                           cipher_11, cipher_12, ct_diff_0212, ctr_diff1,
    #                           ctr_diff1, ctr_diff2], 10)
    return X, Y
