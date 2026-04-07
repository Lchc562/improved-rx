import numpy as np
import matplotlib.pyplot as plt

# type0 = np.load('simeck32_data01_1_4_14r_type0_bit_sensitivity.npy')
# type1 = np.load('simeck32_data01_1_4_14r_type1_bit_sensitivity.npy')

# type0 = np.load('simeck32_data01_1_4_15r_type0_bit_sensitivity.npy')
# type1 = np.load('simeck32_data01_1_4_15r_type1_bit_sensitivity.npy')

# type0 = np.load('simeck32_data01_1_4_16r_type0_bit_sensitivity.npy')
# type1 = np.load('simeck32_data01_1_4_16r_type1_bit_sensitivity.npy')

# type0 = np.load('simeck32_data27_1_4_14111r_type0_bit_sensitivity.npy')
# type1 = np.load('simeck32_data27_1_4_14111r_type1_bit_sensitivity.npy')

# type0 = np.load('simeck32_data27_1_4_15111r_type0_bit_sensitivity.npy')
# type1 = np.load('simeck32_data27_1_4_15111r_type1_bit_sensitivity.npy')

# type0 = np.load('simeck32_data27_1_4_16111r_type0_bit_sensitivity.npy')
# type1 = np.load('simeck32_data27_1_4_16111r_type1_bit_sensitivity.npy')

# type0 = np.load('simeck32_data27_1_4_17111r_type0_bit_sensitivity.npy')
# type1 = np.load('simeck32_data27_1_4_17111r_type1_bit_sensitivity.npy')

type0 = np.load('simeck32_data27_1_4_18111r_type0_bit_sensitivity.npy')
type1 = np.load('simeck32_data27_1_4_18111r_type1_bit_sensitivity.npy')



# print(np.sum(type2[32]-type2))
print(type0)
n = np.shape(type0)
# 横轴
# x = np.linspace(1, n[0], n[0])[:16]
x = np.arange(15, -1, -1)  # 从31到0

# print(x, type0)
width = 0.2

# plt.plot(x1,loss,label = 'train_loss',c = 'r', ls = '-.', marker = 'D', lw = 1)
# c:color ls:lineshape marker:点标记 lw:line_weight
# plt.figure(1)
plt.bar(x - width, type0[16]-type0[:16], width=width, label='KTYPE1')
plt.bar(x, type1[16]-type1[:16], width=width, label='KTYPE2')
# plt.bar(x + width, type2[32]-type2, width=width, label='type2')
plt.xlabel('Bit positon')
plt.ylabel('Bit sensitivity')
# 设置x轴刻度标签
# labels = ['' if i % 5 != 0 else str(i) for i in range(len(x))]  # 仅显示 5 的倍数刻度
plt.xticks(x, range(len(x)), rotation=45)  # 设置刻度标签显示角度为90度
plt.legend()

# 保存为矢量图（PDF格式）
plt.savefig('bit_sensitivity_plot.pdf', format='pdf', bbox_inches='tight')

plt.show()
