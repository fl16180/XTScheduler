import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import math

per_x = 4 / 0.5
per_t = 200 / 0.5
c = 2 * np.pi
xtfunc = lambda x, t: (0.1) * (math.cos(c * (x / per_x + t / per_t)) + 1) / 2 * 0.1 ** (t / 100) + 0.0001


# df = pd.read_csv('./logs/debug/cifar100_resnet18_xts_0.csv')
# acc = df['Acc'].values
# err = np.log(1 - acc)

# fig, ax = plt.subplots(2, 1)

# fig.set_size_inches(4, 4)
# ax[0].plot(range(200), err)

# nt = 200; nx = 4
# mat = np.zeros((nx, nt))
# for t in range(nt):
#     for x in range(nx):
#         mat[x, t] = xtfunc(x, t)

# ax[1].imshow(mat, cmap='Reds', aspect=12, norm=matplotlib.colors.LogNorm())

# ax[0].set_xlim([0, 200])
# ax[1].set_xlim([0, 200])
# ax[0].set_xticks([])
# ax[1].set_yticks([])

# ax[0].set_ylabel('Log error')
# ax[1].set_xlabel('Epoch')

# ax[0].set_title('XTSchedule for Resnet-18 on CIFAR100 ')
# plt.tight_layout()
# plt.savefig('sched.png')


xts = pd.read_csv('./logs/debug/cifar100_resnet20_xts_0.csv')
naive = pd.read_csv('./logs/debug/cifar100_resnet20_naive_0.csv')
cos = pd.read_csv('./logs/debug/cifar100_resnet20_cosine_0.csv')
exp = pd.read_csv('./logs/debug/cifar100_resnet20_exp_0.csv')

t = range(200)
plt.plot(t, xts.Acc.values, label='XTS')
plt.plot(t, naive.Acc.values, label='naive')
plt.plot(t, cos.Acc.values, label='cosine')
plt.plot(t, exp.Acc.values, label='exp decay')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Resnet-20 on CIFAR100')
plt.savefig('cifar100_resnet20.png')







# def show_heatmap(func, nx=10, nt=50, log_color=True):
#     mat = np.zeros((nt, nx))
#     for t in range(nt):
#         for x in range(nx):
#             mat[t, x] = func(x, t)

#     if log_color:
#         plt.imshow(mat, cmap='Reds', aspect=0.15, norm=matplotlib.colors.LogNorm())
#     else:
#         plt.imshow(mat, cmap='Reds', aspect=0.15)

#     fig = plt.gcf()
#     fig.set_size_inches(5, 5)
#     plt.xlabel('layer depth')
#     plt.ylabel('epoch')
#     plt.colorbar(fraction=0.0348, pad=0.04)
#     plt.savefig('tmp.png')