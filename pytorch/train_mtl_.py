import os
import numpy as np
import random
from datetime import datetime
from os import path
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import densenet121
from PIL import Image

# Assuming utils and nets are PyTorch equivalents of the TensorFlow utils and nets
import utils_
from utils_ import *
from nets_ import *

import sys

# Define GPU properties
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

datasetName = sys.argv[1] # Vaihingen, DFC2018

if datasetName == 'DFC2018':
    label_codes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    w1 = 1.0 # sem
    w2 = 1.0 # norm
    w3 = 1.0 # dsm

if datasetName == 'Vaihingen':
    label_codes = [(255, 255, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0)]
    w1 = 1.0 # sem
    w2 = 10.0 # norm
    w3 = 100.0 # dsm

id2code = {k: v for k, v in enumerate(label_codes)}

decay = False
save = True

lr = 0.0002
batchSize = 4
numEpochs = 20
training_samples = 10000
val_freq = 1000
train_iters = int(training_samples / batchSize)
cropSize = 320

predCheckPointPath = './checkpoints/' + datasetName + '/mtl'
corrCheckPointPath = './checkpoints/' + datasetName + '/refinement'

all_rgb, all_dsm, all_sem = collect_tilenames("train", datasetName)
val_rgb, val_dsm, val_sem = collect_tilenames("val", datasetName)

NUM_TRAIN_IMAGES = len(all_rgb)
NUM_VAL_IMAGES = len(val_rgb)

backboneNet = densenet121(pretrained=True)
backboneNet = nn.Sequential(*list(backboneNet.children())[:-1]) # Remove the last layer

net = MTL(backboneNet, datasetName)

min_loss = 1000

for current_epoch in range(1, numEpochs):
    if decay and current_epoch > 1:
        lr = lr / 2
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    print("Current epoch", current_epoch)
    print("Current LR", lr)

    error_ave = 0.0
    error_L1 = 0.0
    error_L2 = 0.0
    error_L3 = 0.0

    for iters in range(train_iters):
        idx = random.randint(0, len(all_rgb) - 1)

        rgb_batch = []
        dsm_batch = []
        sem_batch = []
        norm_batch = []

        if datasetName == 'Vaihingen':
            rgb_tile = np.array(Image.open(all_rgb[idx])) / 255
            dsm_tile = np.array(Image.open(all_dsm[idx])) / 255
            norm_tile = genNormals(dsm_tile)
            sem_tile = np.array(Image.open(all_sem[idx]))

        elif datasetName == 'DFC2018':
            rgb_tile = np.array(Image.open(all_rgb[idx])) / 255
            dsm_tile = np.array(Image.open(all_dsm[2 * idx]))
            dem_tile = np.array(Image.open(all_dsm[2 * idx + 1]))
            dsm_tile = correctTile(dsm_tile)
            dem_tile = correctTile(dem_tile)
            dsm_tile = dsm_tile - dem_tile
            norm_tile = genNormals(dsm_tile)
            sem_tile = np.array(Image.open(all_sem[idx]))

        for i in range(batchSize):
            h = rgb_tile.shape[0]
            w = rgb_tile.shape[1]
            r = random.randint(0, h - cropSize)
            c = random.randint(0, w - cropSize)
            rgb = rgb_tile[r:r + cropSize, c:c + cropSize]
            dsm = dsm_tile[r:r + cropSize, c:c + cropSize]
            sem = sem_tile[r:r + cropSize, c:c + cropSize]
            if datasetName == 'DFC2018':
                sem = sem[..., np.newaxis]
            norm = norm_tile[r:r + cropSize, c:c + cropSize]

            rgb_batch.append(rgb)
            dsm_batch.append(dsm)
            sem_batch.append(rgb_to_onehot(sem, datasetName, id2code))
            norm_batch.append(norm)

        rgb_batch = torch.tensor(rgb_batch, dtype=torch.float32).cuda()
        dsm_batch = torch.tensor(dsm_batch, dtype=torch.float32).cuda()[..., np.newaxis]
        sem_batch = torch.tensor(sem_batch, dtype=torch.float32).cuda()
        norm_batch = torch.tensor(norm_batch, dtype=torch.float32).cuda()

        dsm_out, sem_out, norm_out = net(rgb_batch)
        L1 = nn.MSELoss()(dsm_batch.squeeze(), dsm_out.squeeze())
        L2 = nn.CrossEntropyLoss()(sem_out, sem_batch.argmax(dim=1))
        L3 = nn.MSELoss()(norm_batch, norm_out)
        total_loss = w1 * L2 + w2 * L3 + w3 * L1

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        error_ave += total_loss.item()
        error_L1 += L1.item()
        error_L2 += L2.item()
        error_L3 += L3.item()

        if iters % val_freq == 0 and iters > 0:
            print(iters)
            print('total loss :', error_ave / val_freq)
            print('DSM loss   :', error_L1 / val_freq)
            if net.sem_flag and not net.norm_flag:
                print('SEM loss   :', error_L2 / val_freq)
            if not net.sem_flag and net.norm_flag:
                print('NORM loss   :', error_L3 / val_freq)
            if net.sem_flag and net.norm_flag:
                print('SEM loss   :', error_L2 / val_freq)
                print('NORM loss :', error_L3 / val_freq)

            if error_L1 / val_freq < min_loss and save:
                torch.save(net.state_dict(), predCheckPointPath)
                min_loss = error_L1 / val_freq
                print('dsm train checkpoint saved!')

            error_ave = 0.0
            error_L1 = 0.0
            error_L2 = 0.0
            error_L3 = 0.0

