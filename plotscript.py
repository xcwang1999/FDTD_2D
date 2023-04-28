import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['font.family'] = "Times New Roman"
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['grid.linestyle'] = 'dotted'

grid_row = 200
grid_col = 200

xv, yv = np.meshgrid(range(grid_col), range(grid_row))

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection="3d")

def plot_e_field(data, timestep):
    ax.plot_surface(yv, xv, data, cmap=plt.cm.viridis, edgecolor='black', linewidth=.25)
    ax.view_init(elev=20, azim=35)
    ax.set_box_aspect(None, zoom=1)
    ax.set_zlim(-0.5, 1)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r'$E_{Z}$')
    ax.text2D(0.7, 0.7, "Time step = {}".format(timestep), transform=ax.transAxes)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.set_facecolor('white')


data = []
with open("data_cu.txt", "r") as f:
    lines = f.readlines()

    for i in range(0, len(lines), grid_row):
        data.append(np.loadtxt(lines[i:i+grid_row], delimiter=" "))

print(data[1].shape)
def animate(frame):
    ax.clear()
    plot_e_field(data[frame], frame)

ani = animation.FuncAnimation(fig=fig, func=animate, frames=len(data), interval=20)
ani.save("ez_cu.gif", writer="pillow", )

plt.show()

# timestep = 2000
# timeGPU = []
# with open("execution_time_GPU.txt") as f:
#     timeGPU = np.loadtxt(f)

# timeCPU = []
# with open("execution_time_CPU.txt") as f:
#     timeCPU = np.loadtxt(f)

# fig1 = plt.figure(figsize=(8, 8))
# ax1 = fig1.add_subplot()
# ax1.semilogy(timeGPU, label="GPU")
# ax1.semilogy(timeCPU, label="CPU")
# ax1.set_xlim(0, timestep)
# ax1.set_xlabel("timestep")
# ax1.set_ylabel("execution time(s)")
# ax1.legend()
# ax1.set_title("Time required for each loop")
# plt.savefig("time_comparison.png")
# plt.show()
