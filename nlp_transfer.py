import argparse
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from tqdm import tqdm
from birdtracks import StepTracker

from transformers import BartModel, BartTokenizer, BartConfig, BartForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

from xt_schedule import LambdaXTScheduler

DEVICE = 'cuda'

# the first and second sentence have different keys by dataset
PAIR_KEYS = {
    'rte': ['premise', 'hypothesis'],
    'cb': ['premise', 'hypothesis'],
    'axb': ['sentence1', 'sentence2']
}

parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v')
parser.add_argument('--task', default='rte')
parser.add_argument('--rep', type=int, default=0)
parser.add_argument('--info', type=str, default='')
args = parser.parse_args()

log_dir = './logs/nlp'

if len(args.info) > 0:
    args.info = '_' + args.info
log = StepTracker(name=f'{args.task}_{args.version}_{args.rep}{args.info}',
                    log_dir=log_dir)

KEY1, KEY2 = PAIR_KEYS[args.task]


def train_classifier(classifier, train_loader, tokenizer, optimizer, scheduler):
    classifier.train()
    loss_fn = nn.BCEWithLogitsLoss()

    for batch in tqdm(train_loader):
        X = tokenizer(batch[KEY1], batch[KEY2],
            padding=True, truncation='only_first', return_tensors='pt')

        target = 1 - batch['label']

        X, target = X.to(DEVICE), target.to(DEVICE)
        target = target.float()

        optimizer.zero_grad()
        out = classifier(**X)[0]
        loss = loss_fn(out.flatten(), target)

        loss.backward()
        optimizer.step()


def eval_classifier(classifier, data_loader, tokenizer):
    classifier.eval()
    preds = []
    truth = []

    for batch in tqdm(data_loader):
        X = tokenizer(batch[KEY1], batch[KEY2],
            padding=True, truncation='only_first', return_tensors='pt')

        target = 1 - batch['label']

        X, target = X.to(DEVICE), target.to(DEVICE)
        target = target.float()

        out = classifier(**X)[0]

        preds.append(out.flatten().detach())
        truth.append(target)

    preds = torch.cat(preds)
    truth = torch.cat(truth)

    bin_pred = (preds > 0).float()
    acc = torch.sum(bin_pred == truth) / len(truth.flatten())
    return acc.item()


if __name__ == '__main__':

    if args.task != 'axb':
        dataset = load_dataset('super_glue', args.task)
        train_ds, valid_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']
        print(len(train_ds), len(valid_ds))
    else:
        train_ds = load_dataset('super_glue', args.task, split='test[:85%]')
        valid_ds = load_dataset('super_glue', args.task, split='test[85%:]')
        print(len(train_ds), len(valid_ds))

    train_loader = DataLoader(
        train_ds, batch_size=16, shuffle=True
    )

    valid_loader = DataLoader(
        valid_ds, batch_size=16, shuffle=False
    )

    # use bert because we can easily map an X-function over encoder
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=1)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model.classifier12 = nn.Sequential(
        nn.Linear(768, 768),
        nn.Tanh(),
        nn.Linear(768, 1)
    )

    model = model.to(DEVICE)


    # ---------------- define scheduling versions --------------- #
    class DummySchedule:
        def __init__(self, *args, **kwargs):
            pass
        def step(self):
            pass

    if args.version == 'frozen':
        for param in model.base_model.parameters():
            param.requires_grad = False

        opt = torch.optim.Adam(model.parameters(), lr=2e-5)
        scheduler = DummySchedule()

    if args.version == 'unfrozen':
        opt = torch.optim.Adam(model.parameters(), lr=2e-5)
        scheduler = DummySchedule()

    if args.version == 'linear':
        xtfunc = lambda x, t: (5e-5)/12 * x + 1e-6

        scheduler = LambdaXTScheduler(model, torch.optim.Adam, {'lr': 1e-3}, xt_func=xtfunc, x_template='(\d+).')
        opt = scheduler.get_optimizer()

    if args.version == 'decay':
        c = 2 * np.pi
        xtfunc = lambda x, t: 5e-5 * (-math.cos(c * (x / 24 + t / 20)) + 1) / 2 * 0.1 ** (t / 10)

        scheduler = LambdaXTScheduler(model, torch.optim.Adam, {'lr': 1e-3}, xt_func=xtfunc, x_template='(\d+).')
        opt = scheduler.get_optimizer()


    for epoch in range(10):
        print(epoch)

        train_classifier(model, train_loader, tokenizer, opt, scheduler)
        train_acc = eval_classifier(model, train_loader, tokenizer)
        valid_acc = eval_classifier(model, valid_loader, tokenizer)

        scheduler.step()

        log.add(epoch, 'train_acc', train_acc)
        log.add(epoch, 'valid_acc', valid_acc)
        log.export()
