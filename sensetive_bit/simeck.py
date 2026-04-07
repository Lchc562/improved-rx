import gc

import numpy as np
from os import urandom

a_circle = 5
b_circle = 0
c_circle = 1

# const_simeck32
const_simeck = [0xfffd, 0xfffd, 0xfffd, 0xfffd,
                0xfffd, 0xfffc, 0xfffc, 0xfffc,
                0xfffd, 0xfffd, 0xfffc, 0xfffd,
                0xfffd, 0xfffd, 0xfffc, 0xfffd,
                0xfffc, 0xfffd, 0xfffc, 0xfffc,
                0xfffc, 0xfffc, 0xfffd, 0xfffc,
                0xfffc, 0xfffd, 0xfffc, 0xfffd,
                0xfffd, 0xfffc, 0xfffc, 0xfffd]


def WORD_SIZE():
    return 16


MASK_VAL = 2 ** WORD_SIZE() - 1


def rol(x, k):
    return ((x << k) & MASK_VAL) | (x >> (WORD_SIZE() - k))


def enc_one_round_simeck(p, k):
    c1 = p[0]
    c0 = (rol(p[0], a_circle) & rol(p[0], b_circle)) ^ rol(p[0], c_circle) ^ p[1] ^ k
    return c0, c1


def dec_one_round_simeck(c, k):
    c0 = c[0]
    c1 = c[1]
    p0 = c1
    p1 = (rol(c1, a_circle) & rol(c1, b_circle)) ^ rol(c1, c_circle) ^ c0 ^ k
    return p0, p1


def decrypt_simeck(c, ks):
    x, y = c[0], c[1]
    for k in reversed(ks):
        x, y = dec_one_round_simeck((x, y), k)
    return x, y


def expand_key_simeck(k, t):
    ks = [0 for i in range(t)]
    ks_tmp = [0, 0, 0, 0]
    ks_tmp[0] = k[3]
    ks_tmp[1] = k[2]
    ks_tmp[2] = k[1]
    ks_tmp[3] = k[0]
    ks[0] = ks_tmp[0]
    for i in range(1, t):
        ks[i] = ks_tmp[1]
        tmp = (rol(ks_tmp[1], a_circle) & rol(ks_tmp[1], b_circle)) ^ rol(ks_tmp[1], c_circle) ^ ks[i - 1] ^ \
              const_simeck[i - 1]
        ks_tmp[1] = ks_tmp[2]
        ks_tmp[2] = ks_tmp[3]
        ks_tmp[3] = tmp
    return ks


def encrypt_simeck(p, ks):
    x, y = p[0], p[1]
    for k in ks:
        x, y = enc_one_round_simeck((x, y), k)
    return x, y


def check_testvector_32():
    # key= (0x1918,0x1110,0x0908,0x0100)
    key = (0x1918, 0x1110, 0x0908, 0x0100)
    pt = (0x6565, 0x6877)
    ks = expand_key_simeck(key, 32)
    ct = encrypt_simeck(pt, ks)
    # p  = decrypt(ct, ks)
    if ((ct == (0x770d, 0x2c76))):
        print("Testvector verified.")
        return 1
    else:
        print("Testvector not verified.")
        return 0


def RXDR_convert_to_bits(arr, num_word=4):
    X = np.zeros((num_word * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(num_word * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X


def convert_to_binary(l):
    n = len(l)
    k = WORD_SIZE() * n
    X = np.zeros((k, len(l[0])), dtype=np.uint8)
    for i in range(k):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - 1 - i % WORD_SIZE()
        X[i] = (l[index] >> offset) & 1
    X = X.transpose()
    return X


def make_diff_decry(cts, rounds=1, cycle=None):
    if cycle is None:
        cycle = [5,0,1]
    ct0 = rol(cts[1], cycle[0]) & rol(cts[1], cycle[1]) ^ rol(cts[1], cycle[2]) ^ cts[0]
    ct1 = rol(cts[3], cycle[0]) & rol(cts[3], cycle[1]) ^ rol(cts[3], cycle[2]) ^ cts[2]
    if rounds == 1:
        return ct0 ^ ct1
    if rounds == 2:
        ct0 = rol(ct0, cycle[0]) & rol(ct0, cycle[1]) ^ rol(ct0, cycle[2]) ^ cts[1]
        ct1 = rol(ct1, cycle[0]) & rol(ct1, cycle[1]) ^ rol(ct1, cycle[2]) ^ cts[3]
        return ct0 ^ ct1


def make_train_data(n, nr, pairs=1, k=1, diff=(0x0000, 0x0002)):
    check_testvector_32()
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    Y1 = np.tile(Y, pairs)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys = np.tile(keys, pairs)
    key1 = rol(keys, k)
    key1[3] = key1[3] ^ diff[1]

    ks = expand_key_simeck(keys, nr)
    ks1 = expand_key_simeck(key1, nr)

    plain0l = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16)
    plain1l = rol(plain0l, k) ^ diff[0]
    plain1r = rol(plain0r, k) ^ diff[1]
    num_rand_samples = np.sum(Y1 == 0)

    plain1l[Y1 == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    plain1r[Y1 == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)

    ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks1)
    ctdata0l, ctdata0r = rol(ctdata0l, k), rol(ctdata0r, k)

    print("encrypt_over")
    del plain0l,plain0r,plain1l,plain1r
    gc.collect()

    delta_ctdata0l = ctdata0l ^ ctdata1l
    delta_ctdata0r = ctdata0r ^ ctdata1r
    # print(delta_ctdata0l, delta_ctdata0r)

    secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l
    secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l

    delta_secondLast_ctdata0r = secondLast_ctdata0r ^ secondLast_ctdata1r

    thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r, a_circle) & rol(secondLast_ctdata0r, b_circle) ^ rol(
        secondLast_ctdata0r, c_circle)
    thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r, a_circle) & rol(secondLast_ctdata1r, b_circle) ^ rol(
        secondLast_ctdata1r, c_circle)

    delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r
    X = RXDR_convert_to_bits([delta_ctdata0l, delta_ctdata0r, ctdata0l, ctdata0r, ctdata1l, ctdata1r, delta_secondLast_ctdata0r,
         delta_thirdLast_ctdata0r],8)
    # X = RXDR_convert_to_bits([delta_ctdata0r, delta_secondLast_ctdata0r, delta_thirdLast_ctdata0r],3)
    print("convert_to_binary_over")
    # X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])

    X = X.reshape((pairs, n, WORD_SIZE() * 8)).transpose((1, 0, 2))
    X = X.reshape(n, 1, -1)
    X = np.squeeze(X)
    print("data_over")
    return X, Y


def sensetive_make_train_data(n, nr, k=1, diff=(0x0000, 0x0002)):
    check_testvector_32()
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)

    key1 = rol(keys, k)
    key1[3] = key1[3] ^ diff[1]

    ks = expand_key_simeck(keys, nr)
    ks1 = expand_key_simeck(key1, nr)

    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = rol(plain0l, k) ^ diff[0]
    plain1r = rol(plain0r, k) ^ diff[1]
    num_rand_samples = np.sum(Y == 0)

    plain1l[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    plain1r[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)

    ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks1)
    ctdata0l, ctdata0r = rol(ctdata0l, k), rol(ctdata0r, k)
    demo = [ctdata0l, ctdata0r, ctdata1l, ctdata1r]

    print("encrypt_over")
    del plain0l, plain0r, plain1l, plain1r
    gc.collect()

    delta_ctdata0l = ctdata0l ^ ctdata1l
    delta_ctdata0r = ctdata0r ^ ctdata1r
    # print(delta_ctdata0l, delta_ctdata0r)

    secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l
    secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l

    delta_secondLast_ctdata0r = secondLast_ctdata0r ^ secondLast_ctdata1r

    thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r, a_circle) & rol(secondLast_ctdata0r, b_circle) ^ rol(
        secondLast_ctdata0r, c_circle)
    thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r, a_circle) & rol(secondLast_ctdata1r, b_circle) ^ rol(
        secondLast_ctdata1r, c_circle)

    delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r
    X = RXDR_convert_to_bits(
        [delta_ctdata0l, delta_ctdata0r, ctdata0l, ctdata0r, ctdata1l, ctdata1r, delta_secondLast_ctdata0r,
         delta_thirdLast_ctdata0r], 8)
    # X = RXDR_convert_to_bits([delta_ctdata0r, delta_secondLast_ctdata0r, delta_thirdLast_ctdata0r],3)
    print("convert_to_binary_over")
    # X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X, Y, demo


def make_data_keybst(n, nr, k=1, diff=(0x0000, 0x0002)):
    check_testvector_32()
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)

    key1 = rol(keys, k)
    key1[3] = key1[3] ^ diff[1]

    ks = expand_key_simeck(keys, nr)
    ks1 = expand_key_simeck(key1, nr)
    # print(ks,ks1)

    plain0l = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain1l = rol(plain0l, k) ^ diff[0]
    plain1r = rol(plain0r, k) ^ diff[1]
    num_rand_samples = np.sum(Y == 0)

    plain1l[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    plain1r[Y == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)

    ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks1)
    ctdata0l, ctdata0r = rol(ctdata0l, k), rol(ctdata0r, k)
    demo = [ctdata0l, ctdata0r, ctdata1l, ctdata1r]

    print("encrypt_over")
    del plain0l, plain0r, plain1l, plain1r
    gc.collect()

    delta_ctdata0l = ctdata0l ^ ctdata1l
    delta_ctdata0r = ctdata0r ^ ctdata1r
    # print(delta_ctdata0l, delta_ctdata0r)

    secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l
    secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l

    delta_secondLast_ctdata0r = secondLast_ctdata0r ^ secondLast_ctdata1r

    thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r, a_circle) & rol(secondLast_ctdata0r, b_circle) ^ rol(
        secondLast_ctdata0r, c_circle)
    thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r, a_circle) & rol(secondLast_ctdata1r, b_circle) ^ rol(
        secondLast_ctdata1r, c_circle)

    delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r
    X = RXDR_convert_to_bits(
        [delta_ctdata0l, delta_ctdata0r, ctdata0l, ctdata0r, ctdata1l, ctdata1r, delta_secondLast_ctdata0r,
         delta_thirdLast_ctdata0r], 8)
    # X = RXDR_convert_to_bits([delta_ctdata0r, delta_secondLast_ctdata0r, delta_thirdLast_ctdata0r],3)
    print("convert_to_binary_over")
    # X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X, Y, demo, ks[nr-1], ks1[nr-1]

# a,b,c,d,e = make_data_keybst(4, 5, k=1, diff=(0x0000, 0x0002))
# print(d,e)


def make_data_keybst_pairs(n, nr, k=1, diff=(0x0000, 0x0002), pairs=1):
    print(nr)
    check_testvector_32()
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    Y1 = np.tile(Y, pairs)
    keys = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys = np.tile(keys, pairs)
    key1 = rol(keys, k)
    key1[3] = key1[3] ^ diff[1]

    ks = expand_key_simeck(keys, nr)
    ks1 = expand_key_simeck(key1, nr)
    print(ks,np.array(ks1).shape)

    plain0l = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16)
    plain0r = np.frombuffer(urandom(2 * n * pairs), dtype=np.uint16)
    plain1l = rol(plain0l, k) ^ diff[0]
    plain1r = rol(plain0r, k) ^ diff[1]
    num_rand_samples = np.sum(Y1 == 0)

    plain1l[Y1 == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)
    plain1r[Y1 == 0] = np.frombuffer(urandom(2 * num_rand_samples), dtype=np.uint16)

    ctdata0l, ctdata0r = encrypt_simeck((plain0l, plain0r), ks)
    ctdata1l, ctdata1r = encrypt_simeck((plain1l, plain1r), ks1)
    ctdata0l, ctdata0r = rol(ctdata0l, k), rol(ctdata0r, k)
    demo = [ctdata0l, ctdata0r, ctdata1l, ctdata1r]

    print("encrypt_over")
    del plain0l, plain0r, plain1l, plain1r
    gc.collect()

    delta_ctdata0l = ctdata0l ^ ctdata1l
    delta_ctdata0r = ctdata0r ^ ctdata1r
    # print(delta_ctdata0l, delta_ctdata0r)

    secondLast_ctdata0r = rol(ctdata0r, a_circle) & rol(ctdata0r, b_circle) ^ rol(ctdata0r, c_circle) ^ ctdata0l
    secondLast_ctdata1r = rol(ctdata1r, a_circle) & rol(ctdata1r, b_circle) ^ rol(ctdata1r, c_circle) ^ ctdata1l

    delta_secondLast_ctdata0r = secondLast_ctdata0r ^ secondLast_ctdata1r

    thirdLast_ctdata0r = ctdata0r ^ rol(secondLast_ctdata0r, a_circle) & rol(secondLast_ctdata0r, b_circle) ^ rol(
        secondLast_ctdata0r, c_circle)
    thirdLast_ctdata1r = ctdata1r ^ rol(secondLast_ctdata1r, a_circle) & rol(secondLast_ctdata1r, b_circle) ^ rol(
        secondLast_ctdata1r, c_circle)

    delta_thirdLast_ctdata0r = thirdLast_ctdata0r ^ thirdLast_ctdata1r
    X = RXDR_convert_to_bits([delta_secondLast_ctdata0r, delta_thirdLast_ctdata0r], 2)

    X = X.reshape((pairs, n, WORD_SIZE() * 2)).transpose((1, 0, 2))
    X = X.reshape(n, 1, -1)
    X = np.squeeze(X)
    # X = RXDR_convert_to_bits([delta_ctdata0r, delta_secondLast_ctdata0r, delta_thirdLast_ctdata0r],3)
    print("convert_to_binary_over")
    # X = convert_to_binary([ctdata0l, ctdata0r, ctdata1l, ctdata1r])
    return X, Y, demo, ks[nr-1], ks1[nr-1]
