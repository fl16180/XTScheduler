import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models import resnet

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_ds = torchvision.datasets.CIFAR10('./', train=True, download=True,
                                            transform=transform)
    test_ds = torchvision.datasets.CIFAR10('./', train=False, download=False,
                                            transform=transform)
    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=256, shuffle=True, num_workers=4,
        worker_init_fn=worker_init_fn)

    # also make dev loader

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=4,
        worker_init_fn=worker_init_fn)


    model = load_model('resnet18').to(DEVICE)

    xts = LambdaXTScheduler(model, torch.optim.Adam, {},
        nx=4, nt=50, x_pulses=0.5, t_pulses=3)
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