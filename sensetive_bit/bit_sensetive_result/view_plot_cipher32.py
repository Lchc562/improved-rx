import numpy as np
import matplotlib.pyplot as plt

# type0 = np.load('simeck32_data01_1_4_13r_type0_bit_sensitivity.npy')
# type1 = np.load('simeck32_data01_1_4_13r_type1_bit_sensitivity.npy')
# type2 = np.load('simeck32_data01_1_4_13r_type2_bit_sensitivity.npy')

type0 = np.load('simon32_data02_15_03_11r_type0_bit_sensitivity.npy')
type1 = np.load('simon32_data02_15_03_11r_type1_bit_sensitivity.npy')
type2 = np.load('simon32_data02_15_03_11r_type2_bit_sensitivity.npy')

# type0 = np.load('simon32_data01_3_102_13r_type0_bit_sensitivity.npy')
# type1 = np.load('simon32_data01_3_102_13r_type1_bit_sensitivity.npy')
# type2 = np.load('simon32_data01_3_102_13r_type2_bit_sensitivity.npy')

# type0 = np.load('simon32_data01_15_03_12r_type0_bit_sensitivity.npy')
# type1 = np.load('simon32_data01_15_03_12r_type1_bit_sensitivity.npy')
# type2 = np.load('simon32_data01_15_03_12r_type2_bit_sensitivity.npy')

# type0 = np.load('simon32_data01_15_03_11r_type0_bit_sensitivity.npy')
# type1 = np.load('simon32_data01_15_03_11r_type1_bit_sensitivity.npy')
# type2 = np.load('simon32_data01_15_03_11r_type2_bit_sensitivity.npy')
# print(type0[:32][::-1],type0[:32])

# print(np.sum(type2[32]-type2))
# print(type0)
n = np.shape(type0)
# 横轴

x = np.arange(31, -1, -1)  # 从31到0
# x = np.linspace(1, n[0], n[0])[:32]
# print(x, type0)
width = 0.2

# plt.plot(x1,loss,label = 'train_loss',c = 'r', ls = '-.', marker = 'D', lw = 1)
# c:color ls:lineshape marker:点标记 lw:line_weight
plt.figure(figsize=(6.8, 4.9))
plt.bar(x[:32] - width, type0[32] - type0[:32], width=width, label='TYPE0')
plt.bar(x[:32], type1[32] - type1[:32], width=width, label='TYPE1')
plt.bar(x[:32] + width, type2[32] - type2[:32], width=width, label='TYPE2')
plt.xlabel('Bit position')
plt.ylabel('Bit sensitivity')
# 设置x轴刻度标签
# labels = ['' if i % 5 != 0 else str(i) for i in range(len(x))]  # 仅显示 5 的倍数刻度
plt.xticks(x, range(len(x)), rotation=45)  # 设置刻度标签显示角度为45度
plt.legend(fontsize=7)

# 保存为矢量图（PDF格式）
plt.savefig('bit_sensitivity_plot.pdf', format='pdf', bbox_inches='tight')

# 显示图形
plt.show()