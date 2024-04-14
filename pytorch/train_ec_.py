import os
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.models import densenet121
from nets_ import MTL, Autoencoder
from utils_ import collect_tilenames, correctTile
import torch.optim as optim
import torch.nn.functional as F

import utils_
from nets_ import *
from utils_ import *

# Define GPU properties
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

datasetName = sys.argv[1] # Vaihingen, DFC2018

predCheckPointPath = './checkpoints/' + datasetName + '/mtl'
corrCheckPointPath = './checkpoints/' + datasetName + '/refinement'

save = True

lr = 0.0002
batchSize = 2
numEpochs = 20
training_samples = 10000
val_freq = 1000
cropSize = 320

train_iters = int(training_samples / batchSize)

error_ave = 0.0

all_rgb, all_dsm, all_sem = collect_tilenames("train", datasetName)
val_rgb, val_dsm, val_sem = collect_tilenames("val", datasetName)

print(all_rgb)
print(val_rgb)

NUM_TRAIN_IMAGES = len(all_rgb)
NUM_VAL_IMAGES = len(val_rgb)

print("number of training samples " + str(NUM_TRAIN_IMAGES))
print("number of validation samples " + str(NUM_VAL_IMAGES))

backboneNet = densenet121(pretrained=True)
backboneNet = torch.nn.Sequential(*(list(backboneNet.children())[:-1])) # Remove the last layer

net = MTL(backboneNet, datasetName)
net.load_state_dict(torch.load(predCheckPointPath))
net.eval()

autoencoder = Autoencoder()
optimizer = optim.Adam(autoencoder.parameters(), lr=lr, betas=(0.9, 0.999))

min_loss = 1000

for epoch in range(1, numEpochs):

    print('Current epoch: ' + str(epoch))

    for iters in range(train_iters):

        idx = random.randint(0, len(all_rgb) - 1)

        rgb_batch = []
        dsm_batch = []

        if datasetName == 'Vaihingen':
            rgb_tile = np.array(Image.open(all_rgb[idx])) / 255
            dsm_tile = np.array(Image.open(all_dsm[idx])) / 255

        elif datasetName == 'DFC2018':
            rgb_tile = np.array(Image.open(all_rgb[idx])) / 255
            dsm_tile = np.array(Image.open(all_dsm[2 * idx]))
            dem_tile = np.array(Image.open(all_dsm[2 * idx + 1]))
            dsm_tile = correctTile(dsm_tile)
            dem_tile = correctTile(dem_tile)
            dsm_tile = dsm_tile - dem_tile

        for i in range(batchSize):

            h = rgb_tile.shape[0]
            w = rgb_tile.shape[1]
            r = random.randint(0, h - cropSize)
            c = random.randint(0, w - cropSize)
            rgb = (rgb_tile[r:r + cropSize, c:c + cropSize])
            dsm = (dsm_tile[r:r + cropSize, c:c + cropSize])[..., np.newaxis]

            rgb_batch.append(rgb)
            dsm_batch.append(dsm)

        rgb_batch = np.array(rgb_batch)
        dsm_batch = np.array(dsm_batch)

        dsm_out, sem_out, norm_out = net(torch.from_numpy(rgb_batch).float())
        correctionInput = torch.cat([dsm_out, norm_out, sem_out, torch.from_numpy(rgb_batch).float()], dim=1)

        MSE = torch.nn.MSELoss()

        optimizer.zero_grad()
        noise = autoencoder(correctionInput)
        dsm_out = dsm_out - noise
        total_loss = MSE(dsm_out, torch.from_numpy(dsm_batch).float())
        total_loss.backward()
        optimizer.step()

        error_ave += total_loss.item()

        if iters % val_freq == 0 and iters > 0:

            print(iters)
            print('total loss : ' + str(error_ave / val_freq))

            if error_ave / val_freq < min_loss and save:
                torch.save(autoencoder.state_dict(), corrCheckPointPath)
                min_loss = error_ave / val_freq
                print('train checkpoint saved!')

            error_ave = 0.0

    error_ave = 0.0

