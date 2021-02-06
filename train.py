from resnet import resnet

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data

import torchvision.transforms as transforms

from torch.autograd import Variable

import math
import os

from logger import logger

net = resnet.resnet18(3, 200)
net.cuda()

# loss function + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

# load data set
logger.info("Reading data...")
train_dir = '/media/HDD/datasets/tiny-imagenet-200/train/'
train_dataset = datasets.ImageFolder(train_dir, transform=transforms.ToTensor())
train_loader = data.DataLoader(train_dataset, batch_size=128)
logger.info("Loaded: %s", train_dir)


# train the model
for epoch in range(2):
    logger.info("-- EPOCH: %s", epoch)
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        if i % 50 == 49: 
            logger.info("-- ITERATION: %s", i)
        input, target = data

        # wrap input + target into variables
        input_var = Variable(input).cuda()
        target_var = Variable(target).cuda()
        

        # compute output
        output = net(input_var)
        loss = criterion(output, target_var)
        print(loss.item())

        # computer gradient + sgd step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print progress
        running_loss += loss.item()


        if i % 50 == 49:  # print every 2k mini-batches
            logger.info("-- RUNNING_LOSS: %s", running_loss / 50)
            running_loss = 0.0

logger.info('Finished Training')
torch.save(net.state_dict(), "/models/baseline-resnet18.pt")