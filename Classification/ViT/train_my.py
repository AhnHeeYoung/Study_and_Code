#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import argparse
import os
import random
import numpy as np

from datetime import timedelta

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

from models.modeling import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size

import logging

import torch

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler


# In[ ]:


METHOD = "pretrained_scheduler_epochstep_batchsize41_withouttorchgrad"
run_info = METHOD

if not os.path.exists('checkpoints/{}'.format(run_info)):
    os.mkdir('checkpoints/{}'.format(run_info))

log = logging.getLogger('staining_log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('checkpoints/{}/log.txt'.format(run_info))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
#
log.addHandler(fileHandler)
log.addHandler(streamHandler)


# In[ ]:





# In[2]:


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    num_classes = 10 if args.dataset == "cifar10" else 100

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(num_params)
    return args, model


# In[3]:


config = CONFIGS['ViT-B_16']
num_classes = 10
img_size = 224

model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)


# In[4]:


transform_train = transforms.Compose([
    transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
transform_test = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# In[5]:


trainset = datasets.CIFAR10(root="./data",
                            train=True,
                            download=True,
                            transform=transform_train)

testset = datasets.CIFAR10(root="./data",
                           train=False,
                           download=True,
                           transform=transform_test)


# In[6]:


train_sampler = RandomSampler(trainset)# if args.local_rank == -1 else DistributedSampler(trainset)
test_sampler = SequentialSampler(testset)

train_loader = DataLoader(trainset,
                          sampler=train_sampler,
                          batch_size=4,
                          num_workers=4,
                          pin_memory=True)
test_loader = DataLoader(testset,
                         sampler=test_sampler,
                         batch_size=1,
                         num_workers=4,
                         pin_memory=True) if testset is not None else None


# In[7]:


device='cuda'

model.to(device)
model.load_from(np.load('ViT-B_16.npz'))
log.info('Model Load')


# In[8]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[11]:


optimizer = torch.optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, weight_decay=0)
scheduler = WarmupCosineSchedule(optimizer, warmup_steps=500, t_total=10000)
loss_fct = torch.nn.CrossEntropyLoss()

# In[ ]:

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


num_epochs=100
for epochs in range(num_epochs):
    
    scheduler.step()
    
    train_loss = []
    all_preds, all_label = [], []
    
    model.train()
    for step, batch in enumerate(train_loader):
        
        x = batch[0].to(device)
        y = batch[1].to(device)
        
        output = model(x)[0]
        loss = loss_fct(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        train_loss += [loss.item()]
        preds = torch.argmax(output, -1)
        
        all_preds += preds.tolist()
        all_label += y.tolist()   
        
        acc = simple_accuracy(np.array(all_preds), np.array(all_label))
        
        if step % 1000 == 0:
            log.info("Epoch : {}/{}, Batch : {}/{}, train loss : {:.4f}, accuracy : {:.4f}".format(epochs+1, num_epochs, step+1, len(train_loader),
                                                                          sum(train_loss)/len(train_loss), acc))
            
    log.info("Epoch : {}/{}, Batch : {}/{}, train loss : {:.4f}, accuracy : {:.4f}".format(epochs+1, num_epochs, step+1, len(train_loader),
                                                                          sum(train_loss)/len(train_loss), acc))

            
    test_loss = []
    all_preds, all_label = [], []        
        
    model.eval()
    for step, batch in enumerate(test_loader):
        

        x = batch[0].to(device)
        y = batch[1].to(device)

        output = model(x)[0]
        loss = loss_fct(output, y)

        test_loss += [loss.item()]
        preds = torch.argmax(output, -1)

        all_preds += preds.tolist()
        all_label += y.tolist()   

        acc = simple_accuracy(np.array(all_preds), np.array(all_label))

        if step % 2000 == 0:
            log.info("Epoch : {}/{}, Batch : {}/{}, test loss : {:.4f}, accuracy : {:.4f}".format(epochs+1, num_epochs, step+1, len(test_loader),
                                                                          sum(test_loss)/len(test_loss),
                                                                                          acc))

                
    log.info("Epoch : {}/{}, Batch : {}/{}, test loss : {:.4f}, accuracy : {:.4f}".format(epochs+1, num_epochs, step+1, len(test_loader),
                                                                              sum(test_loss)/len(test_loss),
                                                                                              acc))
    
    torch.save(model.state_dict(), 'checkpoints/{}/Epoch{}TestACC{}'.format(run_info, epochs+1, acc))

    log.info('\n')
    # In[ ]:




