import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models import resnet

import numpy as np
import matplotlib.pyplot as plt

from scheduler import LambdaXTScheduler, CosineXTScheduler

DEVICE = 'cpu'


def load_model(model_name):
    if model_name == 'resnet18':
        model = resnet.resnet18()
    else:
        raise NotImplementedError
    return model


def train(model, train_loader, optimizer, **kwargs):
    ''' train model on train set in train_loader '''
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for data, target in train_loader:
        
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
            out = model(data)
            out = torch.softmax(out, 1).argmax(1)
            preds.append(out.detach())
            truth.append(target)

        preds = torch.cat(preds)
        truth = torch.cat(truth)

    return torch.sum(preds == truth) / len(preds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='resnet18')
    parser.add_argument('--eval', default='dev', choices=['dev', 'test'])
    args = parser.parse_args()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_ds = torchvision.datasets.CIFAR10('./', train=True, download=True,
                                            transform=transform)
    if args.eval == 'dev':
        train_ds, test_ds = torch.utils.data.random_split(train_ds, [50000, 10000])  
    else:
        test_ds = torchvision.datasets.CIFAR10('./', train=False, download=False,
                                            transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=256, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=4


    model = load_model(args.model).to(DEVICE)

    xts = LambdaXTScheduler(model, torch.optim.Adam, {},
        xt_func=lambda x, t: 0.001)
    opt = xts.optimizer


    for epoch in range(50):
        train(model, train_loader, opt)

        acc = eval_model(model, test_loader)
        print('Epoch: ', epoch, 'Acc: ', acc)
        xts.step()


    # xts = CosineXTScheduler(model, torch.optim.Adam, {},
        # nx=4, nt=10, x_pulses=2, t_pulses=2)

    # print(xts.get_group_par_names())
    # opt = xts.optimizer