# MAHDI ELHOUSNI, WPI 2020 
# Altered by Ahmad Naghavi, OzU 2024
# Faithful PyTorch conversion from TensorFlow original

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from config import *


## Utility functions for DenseNet backbone creation and input channel calculation
def calculate_dae_input_channels():
    """Calculate input channels for DAE based on config flags"""
    channels = 1  # DSM output
    channels += 3  # RGB input
    
    if sem_flag:
        channels += sem_k
    if norm_flag:
        channels += 3
    if edge_flag:
        channels += 1
        
    return channels


def create_densenet_backbone(input_channels=3):
    """Create DenseNet121 backbone matching TensorFlow version"""
    if input_channels == 3:
        backbone = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    else:
        # Create DenseNet without pretrained weights for non-standard input channels
        backbone = models.densenet121(weights=None)
        # Modify first conv layer for different input channels
        backbone.features.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Create a feature extractor that returns 4D convolutional features, not 2D pooled features
    # This matches TensorFlow's include_top=False behavior
    class DenseNetFeatures(nn.Module):
        def __init__(self, densenet):
            super().__init__()
            self.features = densenet.features
            
        def forward(self, x):
            features = self.features(x)
            # Apply ReLU and return 4D feature maps (no global pooling)
            return F.relu(features, inplace=True)
    
    return DenseNetFeatures(backbone)


## Multi-task Learning (MTL) architecture - Faithful PyTorch conversion
class MTL(nn.Module):
    def __init__(self, net, sem_flag=True, norm_flag=True, edge_flag=True):
        super(MTL, self).__init__()

        # Determine the initial convolution layer to turn the RGB+SAR 4-channel input to a 3-channel tensor.
        # This is mandatory for the sake of DenseNet121 input as the number of channels shall be three.
        if sar_mode:
            # Define the input shape explicitly as (4, crop_size, crop_size) in PyTorch
            self.conv_sar = nn.Conv2d(4, 3, 3, padding=1)  # TF: Conv2D(3, 3, padding='same', input_shape=(cropSize, cropSize, 4))
            self.do_sar = nn.Dropout(0.5)

        # Determine other model parameters
        self.encoder = net
        self.sem_flag = sem_flag
        self.norm_flag = norm_flag
        self.edge_flag = edge_flag

        self.bn0 = nn.BatchNorm2d(1024)  # Assuming encoder outputs 1024 channels

        # Establish decoding layers for SEM
        if self.sem_flag:
            # TF: Conv2DTranspose(1024, 3, strides=2, padding='same') -> PyTorch: ConvTranspose2d(in_channels, 1024, 3, stride=2, padding=1, output_padding=1)
            self.deconv1_sem = nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1)
            self.deconv2_sem = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
            self.deconv3_sem = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
            self.deconv4_sem = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
            self.deconv5_sem = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

            # Conv2D layers for SEM - TF: Conv2D(1024, 3, strides=1, padding='same') -> PyTorch: Conv2d(in_channels, 1024, 3, stride=1, padding=1)
            self.conv_1_sem = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)  # Will be dynamically calculated based on concatenations
            self.conv_2_sem = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
            self.conv_3_sem = nn.Conv2d(512, 512, 3, stride=1, padding=1)  # Will be dynamically calculated
            self.conv_4_sem = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.conv_5_sem = nn.Conv2d(256, 256, 3, stride=1, padding=1)  # Will be dynamically calculated
            self.conv_6_sem = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_7_sem = nn.Conv2d(64, 64, 3, stride=1, padding=1)    # Will be dynamically calculated
            self.conv_8_sem = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv_9_sem = nn.Conv2d(32, 32, 3, stride=1, padding=1)    # Will be dynamically calculated
            self.conv_10_sem = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            
            self.conv_sem = nn.Conv2d(32, sem_k, 3, stride=1, padding=1)

            self.do1_sem = nn.Dropout(0.5)
            self.do2_sem = nn.Dropout(0.5)
            self.do3_sem = nn.Dropout(0.5)
            self.do4_sem = nn.Dropout(0.5)
            self.do5_sem = nn.Dropout(0.5)

        # Establish decoding layers for Normals
        if self.norm_flag:
            self.deconv1_norm = nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1)
            self.deconv2_norm = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
            self.deconv3_norm = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
            self.deconv4_norm = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
            self.deconv5_norm = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

            self.conv_1_norm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)  # Will be dynamically calculated
            self.conv_2_norm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
            self.conv_3_norm = nn.Conv2d(512, 512, 3, stride=1, padding=1)   # Will be dynamically calculated
            self.conv_4_norm = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.conv_5_norm = nn.Conv2d(256, 256, 3, stride=1, padding=1)   # Will be dynamically calculated
            self.conv_6_norm = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_7_norm = nn.Conv2d(64, 64, 3, stride=1, padding=1)     # Will be dynamically calculated
            self.conv_8_norm = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv_9_norm = nn.Conv2d(32, 32, 3, stride=1, padding=1)     # Will be dynamically calculated
            self.conv_10_norm = nn.Conv2d(32, 32, 3, stride=1, padding=1)

            self.conv_norm = nn.Conv2d(32, 3, 3, stride=1, padding=1)

            self.do1_norm = nn.Dropout(0.5)
            self.do2_norm = nn.Dropout(0.5)
            self.do3_norm = nn.Dropout(0.5)
            self.do4_norm = nn.Dropout(0.5)
            self.do5_norm = nn.Dropout(0.5)

        # Establish decoding layers for EdgeMap
        if self.edge_flag:
            self.deconv1_edge = nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1)
            self.deconv2_edge = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
            self.deconv3_edge = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
            self.deconv4_edge = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
            self.deconv5_edge = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

            self.conv_1_edge = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)  # Will be dynamically calculated
            self.conv_2_edge = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
            self.conv_3_edge = nn.Conv2d(512, 512, 3, stride=1, padding=1)   # Will be dynamically calculated
            self.conv_4_edge = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.conv_5_edge = nn.Conv2d(256, 256, 3, stride=1, padding=1)   # Will be dynamically calculated
            self.conv_6_edge = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_7_edge = nn.Conv2d(64, 64, 3, stride=1, padding=1)     # Will be dynamically calculated
            self.conv_8_edge = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv_9_edge = nn.Conv2d(32, 32, 3, stride=1, padding=1)     # Will be dynamically calculated
            self.conv_10_edge = nn.Conv2d(32, 32, 3, stride=1, padding=1)

            self.conv_edge = nn.Conv2d(32, 1, 3, stride=1, padding=1)

            self.do1_edge = nn.Dropout(0.5)
            self.do2_edge = nn.Dropout(0.5)
            self.do3_edge = nn.Dropout(0.5)
            self.do4_edge = nn.Dropout(0.5)
            self.do5_edge = nn.Dropout(0.5)

        # Establish decoding layers for DSM
        self.deconv1_dsm = nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1)
        self.deconv2_dsm = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv3_dsm = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv4_dsm = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv5_dsm = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)

        # DSM Conv2D layers - input channels will be calculated dynamically based on concatenations
        self.conv_1_dsm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)  # Will be calculated based on concatenated inputs
        self.conv_2_dsm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.conv_3_dsm = nn.Conv2d(512, 512, 3, stride=1, padding=1)    # Will be calculated based on concatenated inputs
        self.conv_4_dsm = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_dsm = nn.Conv2d(256, 256, 3, stride=1, padding=1)    # Will be calculated based on concatenated inputs
        self.conv_6_dsm = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_7_dsm = nn.Conv2d(64, 64, 3, stride=1, padding=1)      # Will be calculated based on concatenated inputs
        self.conv_8_dsm = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_9_dsm = nn.Conv2d(32, 32, 3, stride=1, padding=1)      # Will be calculated based on concatenated inputs
        self.conv_10_dsm = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.conv_dsm = nn.Conv2d(32, 1, 3, stride=1, padding=1)

        self.do1_dsm = nn.Dropout(0.5)
        self.do2_dsm = nn.Dropout(0.5)
        self.do3_dsm = nn.Dropout(0.5)
        self.do4_dsm = nn.Dropout(0.5)
        self.do5_dsm = nn.Dropout(0.5)

        # Fix input channels for concatenated layers based on head flags
        self._fix_concatenation_channels()

    def _fix_concatenation_channels(self):
        """Fix input channels for layers that receive concatenated inputs based on enabled heads"""
        
        # Layer 1 concatenation channels: base + enabled heads
        # DSM layer 1: 1024 (base) + 1024*sem_flag + 1024*norm_flag + 1024*edge_flag
        dsm_layer1_channels = 1024 + (1024 if self.sem_flag else 0) + (1024 if self.norm_flag else 0) + (1024 if self.edge_flag else 0)
        self.conv_1_dsm = nn.Conv2d(dsm_layer1_channels, 1024, 3, stride=1, padding=1)
        
        # SEM/NORM/EDGE layer 1 in call_full: same channels as DSM layer 1
        if self.sem_flag:
            self.conv_1_sem = nn.Conv2d(dsm_layer1_channels, 1024, 3, stride=1, padding=1)
        if self.norm_flag:
            self.conv_1_norm = nn.Conv2d(dsm_layer1_channels, 1024, 3, stride=1, padding=1)
        if self.edge_flag:
            self.conv_1_edge = nn.Conv2d(dsm_layer1_channels, 1024, 3, stride=1, padding=1)

        # Layer 2 concatenation channels: 512 (base) + 512*sem_flag + 512*norm_flag + 512*edge_flag
        dsm_layer2_channels = 512 + (512 if self.sem_flag else 0) + (512 if self.norm_flag else 0) + (512 if self.edge_flag else 0)
        self.conv_3_dsm = nn.Conv2d(dsm_layer2_channels, 512, 3, stride=1, padding=1)
        
        if self.sem_flag:
            self.conv_3_sem = nn.Conv2d(dsm_layer2_channels, 512, 3, stride=1, padding=1)
        if self.norm_flag:
            self.conv_3_norm = nn.Conv2d(dsm_layer2_channels, 512, 3, stride=1, padding=1)
        if self.edge_flag:
            self.conv_3_edge = nn.Conv2d(dsm_layer2_channels, 512, 3, stride=1, padding=1)

        # Layer 3 concatenation channels: 256 (base) + 256*sem_flag + 256*norm_flag + 256*edge_flag
        dsm_layer3_channels = 256 + (256 if self.sem_flag else 0) + (256 if self.norm_flag else 0) + (256 if self.edge_flag else 0)
        self.conv_5_dsm = nn.Conv2d(dsm_layer3_channels, 256, 3, stride=1, padding=1)
        
        if self.sem_flag:
            self.conv_5_sem = nn.Conv2d(dsm_layer3_channels, 256, 3, stride=1, padding=1)
        if self.norm_flag:
            self.conv_5_norm = nn.Conv2d(dsm_layer3_channels, 256, 3, stride=1, padding=1)
        if self.edge_flag:
            self.conv_5_edge = nn.Conv2d(dsm_layer3_channels, 256, 3, stride=1, padding=1)

        # Layer 4 concatenation channels: 64 (base) + 64*sem_flag + 64*norm_flag + 64*edge_flag
        dsm_layer4_channels = 64 + (64 if self.sem_flag else 0) + (64 if self.norm_flag else 0) + (64 if self.edge_flag else 0)
        self.conv_7_dsm = nn.Conv2d(dsm_layer4_channels, 64, 3, stride=1, padding=1)
        
        if self.sem_flag:
            self.conv_7_sem = nn.Conv2d(dsm_layer4_channels, 64, 3, stride=1, padding=1)
        if self.norm_flag:
            self.conv_7_norm = nn.Conv2d(dsm_layer4_channels, 64, 3, stride=1, padding=1)
        if self.edge_flag:
            self.conv_7_edge = nn.Conv2d(dsm_layer4_channels, 64, 3, stride=1, padding=1)

        # Layer 5 concatenation channels: 32 (base) + 32*sem_flag + 32*norm_flag + 32*edge_flag
        dsm_layer5_channels = 32 + (32 if self.sem_flag else 0) + (32 if self.norm_flag else 0) + (32 if self.edge_flag else 0)
        self.conv_9_dsm = nn.Conv2d(dsm_layer5_channels, 32, 3, stride=1, padding=1)
        
        if self.sem_flag:
            self.conv_9_sem = nn.Conv2d(dsm_layer5_channels, 32, 3, stride=1, padding=1)
        if self.norm_flag:
            self.conv_9_norm = nn.Conv2d(dsm_layer5_channels, 32, 3, stride=1, padding=1)
        if self.edge_flag:
            self.conv_9_edge = nn.Conv2d(dsm_layer5_channels, 32, 3, stride=1, padding=1)

    def forward(self, x, head_mode='dsm', training=True):
        # Fuse RGB and SAR inputs if the case
        if sar_mode:
            x = self.conv_sar(x)
            x = self.do_sar(x)  # PyTorch dropout doesn't need training flag

        # Applying the MTL encoder on the input
        x = self.encoder(x)
        x = self.bn0(x)

        # Decoding the encoded input based on the amount of selected interconnection among MTL heads
        if head_mode == 'dsm':
            x_dsm, x_sem, x_norm, x_edge = self.call_dsm(x, training)
        elif head_mode == 'full':
            x_dsm, x_sem, x_norm, x_edge = self.call_full(x, training)
        else:
            # Default fallback to dsm mode
            x_dsm, x_sem, x_norm, x_edge = self.call_dsm(x, training)

        return x_dsm, x_sem, x_norm, x_edge

    def call_dsm(self, x, training=True):
        x_sem, x_norm, x_edge = None, None, None

        # SEM decoding layers
        if self.sem_flag:
            x_sem = self.deconv1_sem(x)
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
            x_sem = F.softmax(x_sem, dim=1)  # TF: axis=3 -> PyTorch: dim=1 (channel dimension)

        # Normal decoding layers
        if self.norm_flag:
            x_norm = self.deconv1_norm(x)
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

        # EdgeMap decoding layers
        if self.edge_flag:
            x_edge = self.deconv1_edge(x)
            x3_edge = F.relu(x_edge)
            x_edge = self.do1_edge(x3_edge)
            x_edge = self.deconv2_edge(x_edge)
            x4_edge = F.relu(x_edge)
            x_edge = self.do2_edge(x4_edge)
            x_edge = self.deconv3_edge(x_edge)
            x5_edge = F.relu(x_edge)
            x_edge = self.do3_edge(x5_edge)
            x_edge = self.deconv4_edge(x_edge)
            x6_edge = F.relu(x_edge)
            x_edge = self.do4_edge(x6_edge)
            x_edge = self.deconv5_edge(x_edge)
            x7_edge = F.relu(x_edge)
            x_edge = self.do5_edge(x7_edge)
            x_edge = self.conv_edge(x_edge)
            x_edge = F.relu(x_edge)

        ####### DSM head decoding layers intertwined with other heads #######
        #### DSM head decoding layer 1 ####
        x_dsm = self.deconv1_dsm(x)
        x_dsm = F.relu(x_dsm)
        concat_list = [x_dsm]
        if self.sem_flag:
            concat_list.append(x3_sem)
        if self.norm_flag:
            concat_list.append(x3_norm)
        if self.edge_flag:
            concat_list.append(x3_edge)
        x_dsm = torch.cat(concat_list, dim=1)  # TF: axis=3 -> PyTorch: dim=1
        x_dsm = self.conv_1_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_2_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do1_dsm(x_dsm)

        #### DSM head decoding layer 2 ####
        x_dsm = self.deconv2_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        concat_list = [x_dsm]
        if self.sem_flag:
            concat_list.append(x4_sem)
        if self.norm_flag:
            concat_list.append(x4_norm)
        if self.edge_flag:
            concat_list.append(x4_edge)
        x_dsm = torch.cat(concat_list, dim=1)
        x_dsm = self.conv_3_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_4_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do2_dsm(x_dsm)

        #### DSM head decoding layer 3 ####
        x_dsm = self.deconv3_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        concat_list = [x_dsm]
        if self.sem_flag:
            concat_list.append(x5_sem)
        if self.norm_flag:
            concat_list.append(x5_norm)
        if self.edge_flag:
            concat_list.append(x5_edge)
        x_dsm = torch.cat(concat_list, dim=1)
        x_dsm = self.conv_5_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_6_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do3_dsm(x_dsm)

        #### DSM head decoding layer 4 ####
        x_dsm = self.deconv4_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        concat_list = [x_dsm]
        if self.sem_flag:
            concat_list.append(x6_sem)
        if self.norm_flag:
            concat_list.append(x6_norm)
        if self.edge_flag:
            concat_list.append(x6_edge)
        x_dsm = torch.cat(concat_list, dim=1)
        x_dsm = self.conv_7_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_8_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do4_dsm(x_dsm)

        #### DSM head decoding layer 5 ####
        x_dsm = self.deconv5_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        concat_list = [x_dsm]
        if self.sem_flag:
            concat_list.append(x7_sem)
        if self.norm_flag:
            concat_list.append(x7_norm)
        if self.edge_flag:
            concat_list.append(x7_edge)
        x_dsm = torch.cat(concat_list, dim=1)
        x_dsm = self.conv_9_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_10_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do5_dsm(x_dsm)

        x_dsm = self.conv_dsm(x_dsm)

        return x_dsm, x_sem, x_norm, x_edge

    def call_full(self, x, training=True):
        x_sem, x_norm, x_edge = None, None, None

        ####### Decoding layers intertwined all with each other #######

        ####    Layer 1     ####
        # DSM head decoding layer 1
        x_dsm = self.deconv1_dsm(x)
        x_concat = F.relu(x_dsm)
        
        if self.sem_flag:
            x_sem = self.deconv1_sem(x)
            x_sem = F.relu(x_sem)
            x_concat = torch.cat([x_concat, x_sem], dim=1)
        if self.norm_flag:
            x_norm = self.deconv1_norm(x)
            x_norm = F.relu(x_norm)
            x_concat = torch.cat([x_concat, x_norm], dim=1)
        if self.edge_flag:
            x_edge = self.deconv1_edge(x)
            x_edge = F.relu(x_edge)
            x_concat = torch.cat([x_concat, x_edge], dim=1)
            
        x_dsm = self.conv_1_dsm(x_concat)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_2_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do1_dsm(x_dsm)

        # SEM head decoding layer 1
        if self.sem_flag:
            x_sem = self.conv_1_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_2_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do1_sem(x_sem)

        # Norm head decoding layer 1
        if self.norm_flag:
            x_norm = self.conv_1_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_2_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do1_norm(x_norm)

        # Edge head decoding layer 1
        if self.edge_flag:
            x_edge = self.conv_1_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_2_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do1_edge(x_edge)

        ####    Layer 2     ####
        # DSM head decoding layer 2
        x_dsm = self.deconv2_dsm(x_dsm)
        x_concat = F.relu(x_dsm)
        
        if self.sem_flag:
            x_sem = self.deconv2_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_concat = torch.cat([x_concat, x_sem], dim=1)
        if self.norm_flag:
            x_norm = self.deconv2_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_concat = torch.cat([x_concat, x_norm], dim=1)
        if self.edge_flag:
            x_edge = self.deconv2_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_concat = torch.cat([x_concat, x_edge], dim=1)
            
        x_dsm = self.conv_3_dsm(x_concat)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_4_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do2_dsm(x_dsm)

        # SEM head decoding layer 2
        if self.sem_flag:
            x_sem = self.conv_3_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_4_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do2_sem(x_sem)

        # Norm head decoding layer 2
        if self.norm_flag:
            x_norm = self.conv_3_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_4_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do2_norm(x_norm)

        # Edge head decoding layer 2
        if self.edge_flag:
            x_edge = self.conv_3_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_4_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do2_edge(x_edge)

        ####    Layer 3    ####
        # DSM head decoding layer 3
        x_dsm = self.deconv3_dsm(x_dsm)
        x_concat = F.relu(x_dsm)
        
        if self.sem_flag:
            x_sem = self.deconv3_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_concat = torch.cat([x_concat, x_sem], dim=1)
        if self.norm_flag:
            x_norm = self.deconv3_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_concat = torch.cat([x_concat, x_norm], dim=1)
        if self.edge_flag:
            x_edge = self.deconv3_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_concat = torch.cat([x_concat, x_edge], dim=1)
            
        x_dsm = self.conv_5_dsm(x_concat)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_6_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do3_dsm(x_dsm)

        # SEM head decoding layer 3
        if self.sem_flag:
            x_sem = self.conv_5_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_6_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do3_sem(x_sem)

        # Norm head decoding layer 3
        if self.norm_flag:
            x_norm = self.conv_5_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_6_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do3_norm(x_norm)

        # Edge head decoding layer 3
        if self.edge_flag:
            x_edge = self.conv_5_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_6_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do3_edge(x_edge)

        ####    Layer 4     ####
        # DSM head decoding layer 4
        x_dsm = self.deconv4_dsm(x_dsm)
        x_concat = F.relu(x_dsm)
        
        if self.sem_flag:
            x_sem = self.deconv4_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_concat = torch.cat([x_concat, x_sem], dim=1)
        if self.norm_flag:
            x_norm = self.deconv4_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_concat = torch.cat([x_concat, x_norm], dim=1)
        if self.edge_flag:
            x_edge = self.deconv4_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_concat = torch.cat([x_concat, x_edge], dim=1)
            
        x_dsm = self.conv_7_dsm(x_concat)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_8_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do4_dsm(x_dsm)

        # SEM head decoding layer 4
        if self.sem_flag:
            x_sem = self.conv_7_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_8_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do4_sem(x_sem)

        # Norm head decoding layer 4
        if self.norm_flag:
            x_norm = self.conv_7_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_8_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do4_norm(x_norm)

        # Edge head decoding layer 4
        if self.edge_flag:
            x_edge = self.conv_7_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_8_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do4_edge(x_edge)

        ####    Layer 5     ####
        # DSM head decoding layer 5
        x_dsm = self.deconv5_dsm(x_dsm)
        x_concat = F.relu(x_dsm)
        
        if self.sem_flag:
            x_sem = self.deconv5_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_concat = torch.cat([x_concat, x_sem], dim=1)
        if self.norm_flag:
            x_norm = self.deconv5_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_concat = torch.cat([x_concat, x_norm], dim=1)
        if self.edge_flag:
            x_edge = self.deconv5_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_concat = torch.cat([x_concat, x_edge], dim=1)
            
        x_dsm = self.conv_9_dsm(x_concat)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_10_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do5_dsm(x_dsm)

        x_dsm = self.conv_dsm(x_dsm)

        # SEM head decoding layer 5
        if self.sem_flag:
            x_sem = self.conv_9_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_10_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do5_sem(x_sem)

            x_sem = self.conv_sem(x_sem)
            x_sem = F.softmax(x_sem, dim=1)

        # Norm head decoding layer 5
        if self.norm_flag:
            x_norm = self.conv_9_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_10_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do5_norm(x_norm)

            x_norm = self.conv_norm(x_norm)

        # Edge head decoding layer 5
        if self.edge_flag:
            x_edge = self.conv_9_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_10_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do5_edge(x_edge)

            x_edge = self.conv_edge(x_edge)
            x_edge = F.relu(x_edge)

        return x_dsm, x_sem, x_norm, x_edge


## Denoising Autoencoder architecture - Faithful PyTorch conversion
class Autoencoder(nn.Module):
    def __init__(self, random_noise_size=100):
        super(Autoencoder, self).__init__()

        # Calculate input channels based on config flags
        # DSM(1) + SEM(sem_k if sem_flag else 0) + NORM(3 if norm_flag else 0) + RGB(3) + EDGE(1 if edge_flag else 0)
        input_channels = 1  # DSM
        if sem_flag:
            input_channels += sem_k
        if norm_flag:
            input_channels += 3
        if edge_flag:
            input_channels += 1
        input_channels += 3  # RGB
        
        # Encoding layers
        self.conv1_0 = nn.Conv2d(input_channels, 64, 3, padding=1)  # TF: Conv2D(64, 3, padding='same')
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # TF: MaxPooling2D(pool_size=(2, 2))

        self.conv2_0 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_0 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_0 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge between encoder and decoder
        self.conv5_0 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5 = nn.Conv2d(1024, 1024, 3, padding=1)

        # Decoding layers
        # TF: Conv2DTranspose(512, 2, strides=2, padding='same') -> PyTorch: ConvTranspose2d(1024, 512, 2, stride=2, padding=0)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0)
        self.conv6_0 = nn.Conv2d(1024, 512, 3, padding=1)  # 512 + 512 (skip connection) = 1024 input channels
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0)
        self.conv7_0 = nn.Conv2d(512, 256, 3, padding=1)  # 256 + 256 (skip connection) = 512 input channels
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0)
        self.conv8_0 = nn.Conv2d(256, 128, 3, padding=1)  # 128 + 128 (skip connection) = 256 input channels
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0)
        self.conv9_0 = nn.Conv2d(128, 64, 3, padding=1)   # 64 + 64 (skip connection) = 128 input channels
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1)

        self.out = nn.Conv2d(64, 1, 3, padding=1)  # Output 1 channel for DSM

        self.do1 = nn.Dropout(0.5)
        self.do2 = nn.Dropout(0.5)
        self.do3 = nn.Dropout(0.5)
        self.do4 = nn.Dropout(0.5)

    def forward(self, x, training=True):
        # Call the encoding layers
        x = self.conv1_0(x)
        x = F.relu(x)
        x = self.conv1(x)
        x_1 = F.relu(x)
        x = self.pool1(x_1)

        x = self.conv2_0(x)
        x = F.relu(x)
        x = self.conv2(x)
        x_2 = F.relu(x)
        x = self.pool2(x_2)

        x = self.conv3_0(x)
        x = F.relu(x)
        x = self.conv3(x)
        x_3 = F.relu(x)
        x = self.pool3(x_3)

        x = self.conv4_0(x)
        x = F.relu(x)
        x = self.conv4(x)
        x_4 = F.relu(x)
        x = self.pool4(x_4)

        # Process the bridge connection
        x = self.conv5_0(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)

        # Call the decoding layers
        x = self.up6(x)
        x = F.relu(x)
        x = torch.cat([x_4, x], dim=1)  # TF: concatenate([x_4, x], axis=3) -> PyTorch: torch.cat([x_4, x], dim=1)
        x = self.conv6_0(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.do1(x)

        x = self.up7(x)
        x = F.relu(x)
        x = torch.cat([x_3, x], dim=1)
        x = self.conv7_0(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.do2(x)

        x = self.up8(x)
        x = F.relu(x)
        x = torch.cat([x_2, x], dim=1)
        x = self.conv8_0(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.do3(x)

        x = self.up9(x)
        x = F.relu(x)
        x = torch.cat([x_1, x], dim=1)
        x = self.conv9_0(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.do4(x)

        x = self.out(x)

        return x
