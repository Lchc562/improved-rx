import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_conv_block(ax, x, y, width, height, num_filters, label):
    ax.add_patch(patches.Rectangle((x, y), width, height, fill=False))
    ax.text(x + width/2, y + height/2, f'Conv1D\nks={3 if num_filters == 64 else 1}\nsame\nfilters={num_filters}',
            ha='center', va='center', fontsize=8)

def draw_dense_block(ax, x, y, width, height, units, label):
    ax.add_patch(patches.Rectangle((x, y), width, height, fill=False))
    ax.text(x + width/2, y + height/2, f'Dense\n{units}',
            ha='center', va='center', fontsize=8)

def draw_flatten_block(ax, x, y, width, height, label):
    ax.add_patch(patches.Rectangle((x, y), width, height, fill=False))
    ax.text(x + width/2, y + height/2, 'Flatten',
            ha='center', va='center', fontsize=8)

def draw_se_block(ax, x, y, width, height, label):
    ax.add_patch(patches.Rectangle((x, y), width, height, fill=False))
    ax.text(x + width/2, y + height/2, 'SE Block',
            ha='center', va='center', fontsize=8)

def draw_residual_block(ax, x, y, width, height, label):
    ax.add_patch(patches.Rectangle((x, y), width, height, fill=False))
    ax.text(x + width/2, y + height/2, 'Residual Block',
            ha='center', va='center', fontsize=8)

def draw_input_output(ax, x, y, width, height, label):
    ax.add_patch(patches.Rectangle((x, y), width, height, fill=False))
    ax.text(x + width/2, y + height/2, label,
            ha='center', va='center', fontsize=8)

fig, ax = plt.subplots(figsize=(12, 6))

# Draw input
draw_input_output(ax, 0, 3, 1, 0.5, 'Input\n(None, 1024)')

# Draw reshape
ax.add_patch(patches.Rectangle((1, 3), 0.5, 0.5, fill=False))
ax.text(1.25, 3.25, 'Reshape\n(None, 8, 128)', ha='center', va='center', fontsize=8)

# Draw first conv block
draw_conv_block(ax, 1.75, 3, 1, 0.5, 64, 'Conv1D ks=1 same filters=64')

# Draw dense blocks
draw_dense_block(ax, 3, 3, 0.5, 0.5, 64, 'Dense')
draw_dense_block(ax, 3.75, 3, 0.5, 0.5, 64, 'Dense')

# Draw residual blocks
for i in range(5):
    draw_residual_block(ax, 4.5 + i*1, 3, 1, 0.5, 'Residual Block')

# Draw flatten and dense
draw_flatten_block(ax, 9.5, 3, 0.5, 0.5, 'Flatten')
draw_dense_block(ax, 10.25, 3, 0.5, 0.5, 128, 'Dense')
draw_dense_block(ax, 11, 3, 0.5, 0.5, 1, 'Dense Sigmoid Output')

# Draw SE blocks
for i in range(2):
    draw_se_block(ax, 2.5 + i*2.5, 2, 2, 0.5, 'SE Block')

# Draw additional conv blocks
draw_conv_block(ax, 2.5, 2, 1, 0.5, 64, 'Conv1D ks=3 same filters=64')
draw_conv_block(ax, 5.5, 2, 1, 0.5, 64, 'Conv1D ks=3 same filters=64')

# Set limits and hide axes
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis('off')

plt.show()