import numpy as np
# def make_keybit_array(sensetive_key):
#     num = len(sensetive_key)
#     array = []
#     for i in range(2 ** num):
#         tmp = 0
#         for j in range(num):
#             tmp += ((i >> j) & 1) << sensetive_key[j]
#         array.append(tmp)
#     array = np.array(array)
#     return array
#
# # right_key_bit = 0x39df
# # sensetive_key_bit = 0xc620
# # sensetive_key = [5, 9, 10, 14, 15]
#
#
# right_key_bit = 0x18cf
# sensetive_key_bit = 0xe730
# sensetive_key = [4, 5, 8, 9, 10, 13, 14, 15]
# tmp = make_keybit_array(sensetive_key)
# print(tmp[[82,115]])
a = np.array([0,3,1,2,1,5])
b = np.array([1,0,2,1,5,4])
a = a+b
print(a,a[a<=3])
