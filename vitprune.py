import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

import os
import argparse

from models.pruning_quantify_vit import ViT, channel_selection
from models.vit_slim import ViT_slim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

percent = 0.3

shared_params = {
    "image_size": 224,
    "patch_size": 16,
    "num_classes": 100,
    "dim": 384,
    "depth": 12,
    "heads": 6,
    "mlp_dim": 1536,
    "dropout": 0.01,
    "emb_dropout": 0.01
}


model = ViT(**shared_params)
model = model.to(device)

model_path = "checkpoint/pruning_quantify_vit-16-ckpt.t7"
print("=> loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['acc']
model.load_state_dict(checkpoint['net'])
print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))

total = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        total += m.indexes.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        size = m.indexes.data.shape[0]
        bn[index:(index+size)] = m.indexes.data.abs().clone()
        index += size


y, i = torch.sort(bn)  ## y是正序列，i是索引序列
thre_index = int(total * percent)
thre = y[thre_index] ##这里得到的是阈值

# print(thre)

pruned = 0
cfg = []  ## 每一层剪枝后的通道数列表
cfg_mask = [] ## 每一层channel_selection的掩码列表
for k, m in enumerate(model.modules()):
    if isinstance(m, channel_selection):
        # print(k)
        # print(m)
        if k in [17, 41, 65, 89, 113, 137, 161, 185, 209, 233, 257, 281]:
            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            thre_ = thre.clone()
            while (torch.sum(mask)%6 !=0):                       # heads
                thre_ = thre_ - 0.0001
                mask = weight_copy.gt(thre_).float().cuda()   
        else:
            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask) ##  获得所有层剪枝的通道的总个数
        m.indexes.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))  ## 附加单层剪枝后的通道数
        cfg_mask.append(mask.clone())   ## 附加单层剪枝掩码
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total

print('pruned_ratio = '+ str(pruned_ratio))
print('Pre-processing Successful!!!!!!!')
# print(cfg)


dataset_path = '../autodl-tmp/data/imagenet100'
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
imagenet_dataset = datasets.ImageFolder(root=dataset_path, transform=val_transform)
train_size = int(0.8 * len(imagenet_dataset))
val_size = len(imagenet_dataset) - train_size

_, testset = random_split(imagenet_dataset, [train_size, val_size])
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=8,pin_memory= True) 



def test(model, testloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

test(model,testloader)
cfg_prune = []
for i in range(len(cfg)):
    if i%2!=0:
        cfg_prune.append([cfg[i-1],cfg[i]])

newmodel = ViT_slim(**shared_params, cfg=cfg_prune)

newmodel.to(device)
# num_parameters = sum([param.nelement() for param in newmodel.parameters()])

newmodel_dict = newmodel.state_dict().copy()

i = 0
newdict = {}
for k,v in model.state_dict().items():
    if 'FFN1.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'FFN1.0.bias' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'to_q' in k or 'to_k' in k or 'to_v' in k:
        # print(k)
        # print(v.size())
        # print('----------')        
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'FFN2.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1
    elif 'to_out.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1

    elif k in newmodel.state_dict():
        newdict[k] = v

newmodel_dict.update(newdict)
newmodel.load_state_dict(newmodel_dict)

torch.save(newmodel.state_dict(), 'pruned.pth')
print('after pruning: ', end=' ')
test(newmodel,testloader)
