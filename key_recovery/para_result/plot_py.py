import matplotlib.pyplot as plt
import numpy as np
len = 16
# y = np.load("simon32mean_data25_12r_pairs28.npy")
# y = np.load("simon32mean_data25_13r_pairs28.npy")
# y = np.load("simon32mean_data25_14r_pairs28.npy")

# y = np.load("simon32std_data25_12r_pairs28.npy")
# y = np.load("simon32std_data25_13r_pairs28.npy")
y = np.load("simon32std_data25_14r_pairs28.npy")


plt.figure(figsize=(12,8))
plt.xticks(np.arange(0,65537,4096))
plt.plot(range(0,2**len), y)
plt.plot(range(0,2**len), np.tile(0.5,2**len),label="0.50",c="red")
# plt.plot(range(0,2**16), np.tile(0.025,2**16),label="0.025",c="red")
plt.xlabel('Difference to real key')
plt.ylabel('Mean response')
plt.ylabel('Std response')
plt.legend()

# 保存为矢量图（PDF格式）
plt.savefig('2d_wrong_mean.pdf', format='pdf', bbox_inches='tight')

plt.show()



