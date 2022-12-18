import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


model = resnet.resnet18()

params = list(model.named_parameters())
[params[x][0] for x in range(len(params))]


param_groups = {
    # 0: 
}




# [{'params': param_groups[i], 'lr': func(i)} for i in range(depth)]


class SimpleFNN(nn.Module):
    def __init__(self):
        super().__init__()

        # need to be defined in ascending x order
        self.alpha = nn.Parameter(torch.tensor([1.0]))
        self.lin0 = nn.Linear(2, 2)
        self.lin1_1 = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10)
        self.lin2 = nn.Linear(10, 5)
        self.bn2 = nn.BatchNorm1d(5)
        self.lin3 = nn.Linear(5, 5)

        self.fnn4 = nn.Sequential(nn.ReLU(), nn.Linear(5, 3), nn.ReLU(), nn.Linear(3, 1))

    def forward(self, X):
        out = self.lin1(X)
        out = self.bn(out)
        out = F.relu(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.lin3(out)
        out = self.fnn4(out)
        return out




# specify joint function
func = lambda x, t: 0.001 * (x+1)

func2 = lambda x, t: 0.001 / 2 ** int(t / 10)

func3 = lambda x, t: 0.001 * (x+1) / 2 ** int(t / 10)

func4 = lambda x, t: (np.cos(2 * np.pi * (x/20 + t/20)) + 1) / 20


def suggest_joint_cosine(nx, nt, lr_min, lr_max, x_pulses=0.5, t_pulses=3):
    # JoCose
    
    per_x = nx / x_pulses       # pulses over x (default to half period)
    per_t = nt / t_pulses       # pulses over t
    
    c = 2 * np.pi

    def joint_func(x, t):
        return (lr_max - lr_min) * (np.cos(c * (x / per_x + t / per_t)) + 1) / 2 + lr_min

    return joint_func

jocose = suggest_joint_cosine(10, 50, 0.0001, 0.1, x_pulses=0.5, t_pulses=2.5)

# specify marginal function
xfunc = lambda x: 0.001 * (x+1)
tfunc = lambda t: 1 / 2 ** int(t / 10)
prod_func = lambda x, t: xfunc(x) * tfunc(t)

def get_prod_func(xfunc, tfunc):
    return lambda x, t: xfunc(x) * tfunc(t)


# def specify(xtfunc=None, xfunc=None, tfunc=None)
    # if xtfunc is not None:
    # else 
    # assert xfunc is not None and tfunc is not None
    # xtfunc = get_prod_func(xfunc, tfunc)



def show_heatmap(func, nx=10, nt=50, log_color=True):
    mat = np.zeros((nt, nx))
    for t in range(nt):
        for x in range(nx):
            mat[t, x] = func(x, t)

    if log_color:
        plt.imshow(mat, cmap='Reds', aspect=0.15, norm=matplotlib.colors.LogNorm())
    else:
        plt.imshow(mat, cmap='Reds', aspect=0.15)

    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    plt.xlabel('layer depth')
    plt.ylabel('epoch')
    plt.colorbar(fraction=0.0348, pad=0.04)
    plt.savefig('tmp.png')


if __name__ == '__main__':

    # show_heatmap(jocose, log_color=False)
    # import sys; sys.exit()

    from scheduler import LambdaXTScheduler, CosineXTScheduler
    model = SimpleFNN()
    optimizer = torch.optim.Adam
    opt_params = {}

    # xts = LambdaXTScheduler(model, optimizer, opt_params,
    #     x_func=lambda x: 0.01 * (x+1), t_func=lambda t: (t+1))

    # xts.get_group_par_names()
    # opt = xts.optimizer
    # # opt.param_groups

    # for i in range(10):
    #     print('t', i)
    #     opt.step()

    #     print(opt.param_groups[0]['lr'])
    #     print(opt.param_groups[1]['lr'])
    #     print(opt.param_groups[2]['lr'])


    #     xts.step()


    xts = CosineXTScheduler(model, optimizer, opt_params,
        nx=4, nt=10, x_pulses=2, t_pulses=2)

    xts.get_group_par_names()
    opt = xts.optimizer
    # opt.param_groups

    for i in range(10):
        print('t', i)
        opt.step()
        opt.step()
        opt.step()


        print(opt.param_groups[0]['lr'])
        print(opt.param_groups[1]['lr'])
        print(opt.param_groups[2]['lr'])
        # print(opt.param_groups[3]['lr'])


        xts.step()


# model = torch.nn.Linear(2, 1)
# optimizer = torch.optim.SGD(model.parameters(), lr=100)
# lambda1 = lambda epoch: 0.65 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


# lrs = []

# for i in range(10):
#     optimizer.step()
#     lrs.append(optimizer.param_groups[0]["lr"])
# #     print("Factor = ", round(0.65 ** i,3)," , Learning Rate = ",round(optimizer.param_groups[0]["lr"],3))
#     scheduler.step()

# plt.plot(range(10),lrs)


    # from pdb import set_trace; set_trace()

    # print(list(model.named_parameters()))



    # print(param_x_groups)

# #[a-z]\d+
