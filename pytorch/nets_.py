import torch
import torch.nn as nn
import torch.nn.functional as F

import utils_


class MTL(nn.Module):
    def __init__(self, net, dataset, sem_flag=True, norm_flag=True):
        super(MTL, self).__init__()

        self.encoder = net
        self.sem_flag = sem_flag
        self.norm_flag = norm_flag

        self.bn0 = nn.BatchNorm2d(
        32
        # net.output_channels
        )

        if dataset == 'DFC2018':
            k = 20
        elif dataset == 'Vaihingen':
            k = 6

        self.deconv1_sem = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_sem = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3_sem = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4_sem = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5_sem = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_sem = nn.Conv2d(32, k, kernel_size=3, padding=1)

        self.do1_sem = nn.Dropout(0.5)
        self.do2_sem = nn.Dropout(0.5)
        self.do3_sem = nn.Dropout(0.5)
        self.do4_sem = nn.Dropout(0.5)
        self.do5_sem = nn.Dropout(0.5)

        self.deconv1_norm = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2_norm = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3_norm = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4_norm = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5_norm = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_norm = nn.Conv2d(32, 3, kernel_size=3, padding=1)

        self.do1_norm = nn.Dropout(0.5)
        self.do2_norm = nn.Dropout(0.5)
        self.do3_norm = nn.Dropout(0.5)
        self.do4_norm = nn.Dropout(0.5)
        self.do5_norm = nn.Dropout(0.5)

        self.deconv1 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_1 = nn.Conv2d(32, 1024, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_7 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv_8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv_9 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_10 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv_dsm = nn.Conv2d(32, 1, kernel_size=3, padding=1)

        self.do1 = nn.Dropout(0.5)
        self.do2 = nn.Dropout(0.5)
        self.do3 = nn.Dropout(0.5)
        self.do4 = nn.Dropout(0.5)
        self.do5 = nn.Dropout(0.5)

    def forward(self, x):
        x0 = self.encoder(x)
        x0 = self.bn0(x0)

        if self.sem_flag:
            x_sem = self.deconv1_sem(x0)
            x3_sem = F.relu(x_sem)
            x_sem = self.do1_sem(x3_sem)
            x_sem = self.deconv2_sem(x_sem)
            x4_sem = F.relu(x_sem)
            x_sem = self.do2_sem(x4_sem)
            x_sem = self.deconv3_sem(x_sem)
            x5_sem = F.relu(x_sem)
            x_sem = self.do3_sem(x5_sem)
            x_sem = self.deconv4_sem(x_sem)
            x6_sem = F.relu(x_sem)
            x_sem = self.do4_sem(x6_sem)
            x_sem = self.deconv5_sem(x_sem)
            x7_sem = F.relu(x_sem)
            x_sem = self.do5_sem(x7_sem)
            x_sem = self.conv_sem(x_sem)
            x_sem = F.softmax(x_sem, dim=1)

        if self.norm_flag:
            x_norm = self.deconv1_norm(x0)
            x3_norm = F.relu(x_norm)
            x_norm = self.do1_norm(x3_norm)
            x_norm = self.deconv2_norm(x_norm)
            x4_norm = F.relu(x_norm)
            x_norm = self.do2_norm(x4_norm)
            x_norm = self.deconv3_norm(x_norm)
            x5_norm = F.relu(x_norm)
            x_norm = self.do3_norm(x5_norm)
            x_norm = self.deconv4_norm(x_norm)
            x6_norm = F.relu(x_norm)
            x_norm = self.do4_norm(x6_norm)
            x_norm = self.deconv5_norm(x_norm)
            x7_norm = F.relu(x_norm)
            x_norm = self.do5_norm(x7_norm)
            x_norm = self.conv_norm(x_norm)
            x_norm = F.relu(x_norm)

        x = self.deconv1(x0)
        x = F.relu(x)
        if self.norm_flag:
            x = torch.cat([x, x3_norm], dim=1)
        if self.sem_flag:
            x = torch.cat([x, x3_sem], dim=1)
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.do1(x)

        x = self.deconv2(x)
        x = F.relu(x)
        if self.norm_flag:
            x = torch.cat([x, x4_norm], dim=1)
        if self.sem_flag:
            x = torch.cat([x, x4_sem], dim=1)
        x = self.conv_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = F.relu(x)
        x = self.do2(x)

        x = self.deconv3(x)
        x = F.relu(x)
        if self.norm_flag:
            x = torch.cat([x, x5_norm], dim=1)
        if self.sem_flag:
            x = torch.cat([x, x5_sem], dim=1)
        x = self.conv_5(x)
        x = F.relu(x)
        x = self.conv_6(x)
        x = F.relu(x)
        x = self.do3(x)

        x = self.deconv4(x)
        x = F.relu(x)
        if self.norm_flag:
            x = torch.cat([x, x6_norm], dim=1)
        if self.sem_flag:
            x = torch.cat([x, x6_sem], dim=1)
        x = self.conv_7(x)
        x = F.relu(x)
        x = self.conv_8(x)
        x = F.relu(x)
        x = self.do4(x)

        x = self.deconv5(x)
        x = F.relu(x)
        if self.norm_flag:
            x = torch.cat([x, x7_norm], dim=1)
        if self.sem_flag:
            x = torch.cat([x, x7_sem], dim=1)
        x = self.conv_9(x)
        x = F.relu(x)
        x = self.conv_10(x)
        x = F.relu(x)
        x = self.do5(x)

        x = self.conv_dsm(x)

        return x, x_sem, x_norm



class Autoencoder(nn.Module):
    def __init__(self, random_noise_size=100):
        super(Autoencoder, self).__init__()

        self.conv1_0 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_0 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_0 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_0 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_0 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)

        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.conv6_0 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.conv7_0 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.conv8_0 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.conv9_0 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.out = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.do1 = nn.Dropout(0.5)
        self.do2 = nn.Dropout(0.5)
        self.do3 = nn.Dropout(0.5)
        self.do4 = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.conv1_0(x)
        x1 = F.relu(x1)
        x1 = self.conv1(x1)
        x1 = F.relu(x1)
        x1 = self.pool1(x1)

        x2 = self.conv2_0(x1)
        x2 = F.relu(x2)
        x2 = self.conv2(x2)
        x2 = F.relu(x2)
        x2 = self.pool2(x2)

        x3 = self.conv3_0(x2)
        x3 = F.relu(x3)
        x3 = self.conv3(x3)
        x3 = F.relu(x3)
        x3 = self.pool3(x3)

        x4 = self.conv4_0(x3)
        x4 = F.relu(x4)
        x4 = self.conv4(x4)
        x4 = F.relu(x4)
        x4 = self.pool4(x4)

        x5 = self.conv5_0(x4)
        x5 = F.relu(x5)
        x5 = self.conv5(x5)
        x5 = F.relu(x5)

        x6 = self.up6(x5)
        x6 = F.relu(x6)
        x6 = torch.cat([x4, x6], dim=1)
        x6 = self.conv6_0(x6)
        x6 = F.relu(x6)
        x6 = self.conv6(x6)
        x6 = F.relu(x6)
        x6 = self.do1(x6)

        x7 = self.up7(x6)
        x7 = F.relu(x7)
        x7 = torch.cat([x3, x7], dim=1)
        x7 = self.conv7_0(x7)
        x7 = F.relu(x7)
        x7 = self.conv7(x7)
        x7 = F.relu(x7)
        x7 = self.do2(x7)

        x8 = self.up8(x7)
        x8 = F.relu(x8)
        x8 = torch.cat([x2, x8], dim=1)
        x8 = self.conv8_0(x8)
        x8 = F.relu(x8)
        x8 = self.conv8(x8)
        x8 = F.relu(x8)
        x8 = self.do3(x8)

        x9 = self.up9(x8)
        x9 = F.relu(x9)
        x9 = torch.cat([x1, x9], dim=1)
        x9 = self.conv9_0(x9)
        x9 = F.relu(x9)
        x9 = self.conv9(x9)
        x9 = F.relu(x9)
        x9 = self.do4(x9)

        x = self.out(x9)

        return x

