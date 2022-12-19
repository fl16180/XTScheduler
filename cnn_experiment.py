import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from models.resnet import ResNet18
from models.resnet_s import resnet20

import numpy as np
import matplotlib.pyplot as plt

from xt_schedule import LambdaXTScheduler, CosineXTScheduler
from birdtracks import StepTracker

DEVICE = 'cuda'


def load_model(model_name):
    if model_name == 'resnet18':
        model = ResNet18()
    elif model_name == 'resnet20':
        model = resnet20()
    else:
        raise NotImplementedError
    return model


def train(model, train_loader, optimizer, **kwargs):
    ''' train model on train set in train_loader '''
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for data, target in tqdm(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        out = model(data)
        loss = loss_fn(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_model(model, data_loader, **kwargs):
    ''' eval model on test set in data_loader '''
    model.eval()
    with torch.no_grad():
        preds = []
        truth = []
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            out = model(data)
            out = torch.softmax(out, 1).argmax(1)
            preds.append(out.detach())
            truth.append(target)

        preds = torch.cat(preds)
        truth = torch.cat(truth)

    return torch.sum(preds == truth) / len(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', default='cifar10')
    parser.add_argument('--model', '-m', default='resnet18')
    parser.add_argument('--version', '-v', default='cosine')
    parser.add_argument('--rep', '-r', type=int, default=0)
    parser.add_argument('--eval', default='dev', choices=['dev', 'test'])
    parser.add_argument('--label', default='debug')
    parser.add_argument('--info', default='todo')
    args = parser.parse_args()

    log_name = f'{args.dataset}_{args.model}_{args.version}_{args.rep}'
    tracker = StepTracker(name=log_name, log_dir=f'./logs/{args.label}')
    tracker.annotate(args.info)

    # -------- handle data loading -------- #
    if args.dataset == 'cifar10':
        Dataset = torchvision.datasets.CIFAR10
    elif args.dataset == 'cifar100':
        Dataset = torchvision.datasets.CIFAR100

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_ds = Dataset('./', train=True, download=True,
                                            transform=train_transform)
    if args.eval == 'dev':
        train_ds, test_ds = torch.utils.data.random_split(train_ds, [45000, 5000])  
    else:
        test_ds = Dataset('./', train=False, download=False,
                                            transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=128, shuffle=True, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=128, shuffle=False, num_workers=2)

    # --------- Load model ---------- #
    model = load_model(args.model).to(DEVICE)



    # --------- Handle optimization options ---------- #
    class DummySchedule:
        def __init__(self, *args, **kwargs):
            pass
        def step(self):
            pass

    if args.version == 'naive':

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = DummySchedule()

    if args.version == 'exp':
        xtfunc = lambda x, t: 0.05 * 0.1 ** (t / 100)

        scheduler = LambdaXTScheduler(model, torch.optim.SGD, {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},
            xt_func=xtfunc)
        # scheduler = CosineXTScheduler(model, torch.optim.SGD, {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},
        #     nx=4, nt=200, lr_min=1e-5, lr_max=0.1, x_pulses=0.5, t_pulses=1)
        opt = scheduler.get_optimizer()

    if args.version == 'cosine':

        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)

    elif args.version == 'xts':
        # def sched(x, t):
        #     if t < 60:
        #         lr = 0.1
        #     elif t < 120:
        #         lr = 0.1 * 0.2
        #     elif t < 160:
        #         lr = 0.1 * 0.2 * 0.2
        #     return lr
        import math
        per_x = 4 / 0.5
        per_t = 200 / 0.5
        c = 2 * np.pi
        xtfunc = lambda x, t: (0.1) * (math.cos(c * (x / per_x + t / per_t)) + 1) / 2 * 0.1 ** (t / 100)

        scheduler = LambdaXTScheduler(model, torch.optim.SGD, {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},
            xt_func=xtfunc)
        # scheduler = CosineXTScheduler(model, torch.optim.SGD, {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4},
        #     nx=4, nt=200, lr_min=1e-5, lr_max=0.1, x_pulses=0.5, t_pulses=1)
        opt = scheduler.get_optimizer()


    for epoch in range(200):
        # print(opt.param_groups[0]['lr'])

        train(model, train_loader, opt)
        acc = eval_model(model, test_loader)
        print('Epoch: ', epoch, 'Acc: ', acc.item())
        scheduler.step()

        tracker.add(epoch, 'Acc', acc.item())
        tracker.export()

    # xts = CosineXTScheduler(model, torch.optim.Adam, {},
        # nx=4, nt=10, x_pulses=2, t_pulses=2)

    # print(xts.get_group_par_names())
    # opt = xts.optimizer