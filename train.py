# -*- coding: utf-8 -*-
'''

Train imagenet100 with PyTorch and Vision Transformers!

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import os
import argparse
import pandas as pd
import csv

from models.pruning_quantify_vit import ViT as pruning_quantify_vit

from models.quantify_ALL import ViT as quantify_ALL
from models.quantify_CLS_vit import ViT as quantify_CLS_vit
from models.quantify_FNN_vit import ViT as quantify_FNN_vit
from models.quantify_MHA_vit import ViT as quantify_MHA_vit

from models.quantify_head_vit import ViT as quantify_head_vit
from models.quantify_layer_vit import ViT as quantify_layer_vit

from models.quantify_weight_activation import ViT as quantify_weight_activation
from models.quantify_weight_only import ViT as quantify_weight_only

# from models.pruning_quantify_vit import ViT as pruning_quantify_vit
from torch.nn.utils import clip_grad_value_
from models.vit import ViT as vit
from models.pruning_quantify_vit import channel_selection
from utils import progress_bar, count_MemorySize_Params_FLOPs
from torch.utils.tensorboard import SummaryWriter



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# parsers
parser = argparse.ArgumentParser(description='PyTorch imagenet100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', default='64')
parser.add_argument('--n_epochs', type=int, default='100')
parser.add_argument('--patch', default='16', type=int)
parser.add_argument('--cos', action='store_true', help='Train with cosine annealing scheduling')
parser.add_argument("--bit", type=int, help="bit number for weight each parameter", choices=[8,4,2], default=8)
args = parser.parse_args()

if args.cos:
    from warmup_scheduler import GradualWarmupScheduler

bs = int(args.bs)
k = int(args.bit)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')

dataset_path = '../autodl-tmp/data/imagenet100'

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224), 
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # 验证集不使用数据增强，只进行必要的预处理
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(root=dataset_path, transform=train_transform)
val_dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(val_dataset) - train_size
train_dataset, _ = random_split(train_dataset, [train_size, len(train_dataset) - train_size])
_, val_dataset = random_split(val_dataset, [train_size, len(val_dataset) - train_size])

trainloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8,pin_memory = True)
testloader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=8,pin_memory= True)



# Model
print('==> Building model..')

shared_params = {
    "image_size": 224,
    "patch_size": args.patch,
    "num_classes": 100,
    "dim": 384,
    "depth": 12,
    "heads": 6,
    "mlp_dim": 1536,
    "dropout": 0.01,
    "emb_dropout": 0.01
}

L1 = False
msa = True

if args.net=='quantify_ALL':
    net = quantify_ALL(**shared_params)    
elif args.net=='quantify_CLS_vit':
    net = quantify_CLS_vit(**shared_params)    
elif args.net=='quantify_FNN_vit':
    net = quantify_FNN_vit(**shared_params)    
elif args.net=='quantify_MHA_vit':
    net = quantify_MHA_vit(**shared_params)  
    
elif args.net=='quantify_head_vit':
    net = quantify_head_vit(**shared_params)
elif args.net=='quantify_layer_vit':
    net = quantify_layer_vit(**shared_params)  
    
elif args.net=='quantify_weight_activation':
    net = quantify_weight_activation(**shared_params)    
elif args.net=='quantify_weight_only':
    net = quantify_weight_only(**shared_params) 

elif args.net=='pruning_quantify_vit':
    net = pruning_quantify_vit(**shared_params)    
    L1 = True
elif args.net=="vit":
    # ViT for cifar10
    net = vit(**shared_params)
    msa = False
    
cudnn.benchmark = True
net = net.to(device)

# net.load_state_dict(torch.load(r'./pretrain/S_16_model_parameters.pth'))
pretrain = torch.load(r'./pretrain/pruning_quantify_vit-S16-ckpt.t7')
net.load_state_dict(pretrain['net'], strict=False)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-16-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'],  strict=False)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    
# Loss is CE
criterion = nn.CrossEntropyLoss()
# reduce LR on Plateau
if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)    
elif args.opt == "adamw":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=5e-4)

if not args.cos:
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
else:
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)
    
def sparse_selection():
    s = 1e-4
    for m in net.modules():
        if isinstance(m, channel_selection):
            m.indexes.grad.data.add_(s*torch.sign(m.indexes.data))  # L1            

            
          
            
##### Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        if L1:
            sparse_selection()
        clip_grad_value_(net.parameters(), clip_value=0.1)
               
        optimizer.step()
        if msa:
            net.apply_ema(k)
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
import time
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc

list_loss = []
list_acc = []
for epoch in range(start_epoch, args.n_epochs):
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    if args.cos:
        scheduler.step(epoch-1)
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # write as csv for analysis
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['Loss', 'Accuracy'])
        for loss, acc in zip(list_loss, list_acc):
            writer.writerow([loss, acc])

        
dsize = (1, 3, 224, 224)
inputs = torch.randn(dsize).to(device)
memory_size, params, flops = count_MemorySize_Params_FLOPs(net , inputs)

print('Net:',args.net)
print('MemorySize:',memory_size)
print('Paras:',params)
print('FLOPs:',flops)
