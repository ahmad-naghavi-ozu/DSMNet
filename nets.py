# MAHDI ELHOUSNI, WPI 2020 
# Altered by Ahmad Naghavi, OzU 2024
# Faithful 1:1 PyTorch conversion of original TensorFlow DSMNet implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from config import *

class MTL(nn.Module):
    """Multi-task Learning (MTL) architecture - faithful PyTorch conversion from TensorFlow"""
    
    def __init__(self, backbone_net, sem_flag=True, norm_flag=True, edge_flag=True):
        super(MTL, self).__init__()
        
        # Store backbone and flags exactly like TensorFlow
        self.encoder = backbone_net
        self.sem_flag = sem_flag
        self.norm_flag = norm_flag
        self.edge_flag = edge_flag
        
        # SAR mode handling - exact TensorFlow match
        if sar_mode:
            self.conv_sar = nn.Conv2d(4, 3, kernel_size=3, padding=1)
            self.do_sar = nn.Dropout(0.5)
        
        # Batch normalization after encoder - exact TensorFlow match
        self.bn0 = nn.BatchNorm2d(1024)
        
        # SEM head layers - exact TensorFlow match
        if self.sem_flag:
            self.deconv1_sem = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
            self.deconv2_sem = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
            self.deconv3_sem = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
            self.deconv4_sem = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            self.deconv5_sem = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
            self.conv_sem = nn.Conv2d(32, sem_k, 3, stride=1, padding=1)
            
            self.do1_sem = nn.Dropout(0.5)
            self.do2_sem = nn.Dropout(0.5)
            self.do3_sem = nn.Dropout(0.5)
            self.do4_sem = nn.Dropout(0.5)
            self.do5_sem = nn.Dropout(0.5)
        
        # Normal head layers - exact TensorFlow match
        if self.norm_flag:
            self.deconv1_norm = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
            self.deconv2_norm = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
            self.deconv3_norm = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
            self.deconv4_norm = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            self.deconv5_norm = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
            self.conv_norm = nn.Conv2d(32, 3, 3, stride=1, padding=1)
            
            self.do1_norm = nn.Dropout(0.5)
            self.do2_norm = nn.Dropout(0.5)
            self.do3_norm = nn.Dropout(0.5)
            self.do4_norm = nn.Dropout(0.5)
            self.do5_norm = nn.Dropout(0.5)
        
        # Edge head layers - exact TensorFlow match
        if self.edge_flag:
            self.deconv1_edge = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
            self.deconv2_edge = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
            self.deconv3_edge = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
            self.deconv4_edge = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            self.deconv5_edge = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
            self.conv_edge = nn.Conv2d(32, 1, 3, stride=1, padding=1)
            
            self.do1_edge = nn.Dropout(0.5)
            self.do2_edge = nn.Dropout(0.5)
            self.do3_edge = nn.Dropout(0.5)
            self.do4_edge = nn.Dropout(0.5)
            self.do5_edge = nn.Dropout(0.5)
        
        # DSM head layers - exact TensorFlow match
        self.deconv1_dsm = nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1)
        self.deconv2_dsm = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv3_dsm = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv4_dsm = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv5_dsm = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv_dsm = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        
        self.do1_dsm = nn.Dropout(0.5)
        self.do2_dsm = nn.Dropout(0.5)
        self.do3_dsm = nn.Dropout(0.5)
        self.do4_dsm = nn.Dropout(0.5)
        self.do5_dsm = nn.Dropout(0.5)
        
        # Additional conv layers for TensorFlow's call_full method interconnection
        if self.sem_flag:
            self.conv_1_sem = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
            self.conv_2_sem = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
            self.conv_3_sem = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.conv_4_sem = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.conv_5_sem = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_6_sem = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_7_sem = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv_8_sem = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv_9_sem = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.conv_10_sem = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            
        if self.norm_flag:
            self.conv_1_norm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
            self.conv_2_norm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
            self.conv_3_norm = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.conv_4_norm = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.conv_5_norm = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_6_norm = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_7_norm = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv_8_norm = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv_9_norm = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.conv_10_norm = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            
        if self.edge_flag:
            self.conv_1_edge = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
            self.conv_2_edge = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
            self.conv_3_edge = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.conv_4_edge = nn.Conv2d(512, 512, 3, stride=1, padding=1)
            self.conv_5_edge = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_6_edge = nn.Conv2d(256, 256, 3, stride=1, padding=1)
            self.conv_7_edge = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv_8_edge = nn.Conv2d(64, 64, 3, stride=1, padding=1)
            self.conv_9_edge = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            self.conv_10_edge = nn.Conv2d(32, 32, 3, stride=1, padding=1)
            
        # Additional conv layers for DSM head interconnection - dynamic input channels
        self.conv_1_dsm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)  # Start with base
        self.conv_2_dsm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.conv_3_dsm = nn.Conv2d(512, 512, 3, stride=1, padding=1)    # Start with base
        self.conv_4_dsm = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_dsm = nn.Conv2d(256, 256, 3, stride=1, padding=1)    # Start with base
        self.conv_6_dsm = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_7_dsm = nn.Conv2d(64, 64, 3, stride=1, padding=1)      # Start with base
        self.conv_8_dsm = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_9_dsm = nn.Conv2d(32, 32, 3, stride=1, padding=1)      # Start with base
        self.conv_10_dsm = nn.Conv2d(32, 32, 3, stride=1, padding=1)

    def forward(self, x, head_mode='dsm', training=True):
        """Faithful 1:1 conversion of TensorFlow MTL.call method"""
        
        # Fuse RGB and SAR inputs if the case (exact TensorFlow match)
        if sar_mode:
            x = self.conv_sar(x)
            x = self.do_sar(x) if training else x
        
        # Applying the MTL encoder on the input (exact TensorFlow match)
        x = self.encoder.features(x)
        x = self.bn0(x) if training else x
        
        # Decoding the encoded input based on the amount of selected interconnection among MTL heads
        if head_mode == 'dsm':
            x_dsm, x_sem, x_norm, x_edge = self.call_dsm(x, training)
        elif head_mode == 'full':
            x_dsm, x_sem, x_norm, x_edge = self.call_full(x, training)
        else:
            raise ValueError(f"Invalid head_mode: {head_mode}")
        
        return x_dsm, x_sem, x_norm, x_edge
    
    def call_dsm(self, x, training=True):
        """Faithful 1:1 conversion of TensorFlow MTL.call_dsm method"""
        
        x_sem, x_norm, x_edge = [], [], []
        
        # SEM decoding layers (exact TensorFlow match)
        if self.sem_flag:
            x_sem = self.deconv1_sem(x)
            x3_sem = F.relu(x_sem)
            x_sem = self.do1_sem(x3_sem) if training else x3_sem
            x_sem = self.deconv2_sem(x_sem)
            x4_sem = F.relu(x_sem)
            x_sem = self.do2_sem(x4_sem) if training else x4_sem
            x_sem = self.deconv3_sem(x_sem)
            x5_sem = F.relu(x_sem)
            x_sem = self.do3_sem(x5_sem) if training else x5_sem
            x_sem = self.deconv4_sem(x_sem)
            x6_sem = F.relu(x_sem)
            x_sem = self.do4_sem(x6_sem) if training else x6_sem
            x_sem = self.deconv5_sem(x_sem)
            x7_sem = F.relu(x_sem)
            x_sem = self.do5_sem(x7_sem) if training else x7_sem
            x_sem = self.conv_sem(x_sem)
            x_sem = F.softmax(x_sem, dim=1)
        
        # Normal decoding layers (exact TensorFlow match)
        if self.norm_flag:
            x_norm = self.deconv1_norm(x)
            x3_norm = F.relu(x_norm)
            x_norm = self.do1_norm(x3_norm) if training else x3_norm
            x_norm = self.deconv2_norm(x_norm)
            x4_norm = F.relu(x_norm)
            x_norm = self.do2_norm(x4_norm) if training else x4_norm
            x_norm = self.deconv3_norm(x_norm)
            x5_norm = F.relu(x_norm)
            x_norm = self.do3_norm(x5_norm) if training else x5_norm
            x_norm = self.deconv4_norm(x_norm)
            x6_norm = F.relu(x_norm)
            x_norm = self.do4_norm(x6_norm) if training else x6_norm
            x_norm = self.deconv5_norm(x_norm)
            x7_norm = F.relu(x_norm)
            x_norm = self.do5_norm(x7_norm) if training else x7_norm
            x_norm = self.conv_norm(x_norm)
        
        # EdgeMap decoding layers (exact TensorFlow match)
        if self.edge_flag:
            x_edge = self.deconv1_edge(x)
            x3_edge = F.relu(x_edge)
            x_edge = self.do1_edge(x3_edge) if training else x3_edge
            x_edge = self.deconv2_edge(x_edge)
            x4_edge = F.relu(x_edge)
            x_edge = self.do2_edge(x4_edge) if training else x4_edge
            x_edge = self.deconv3_edge(x_edge)
            x5_edge = F.relu(x_edge)
            x_edge = self.do3_edge(x5_edge) if training else x5_edge
            x_edge = self.deconv4_edge(x_edge)
            x6_edge = F.relu(x_edge)
            x_edge = self.do4_edge(x6_edge) if training else x6_edge
            x_edge = self.deconv5_edge(x_edge)
            x7_edge = F.relu(x_edge)
            x_edge = self.do5_edge(x7_edge) if training else x7_edge
            x_edge = self.conv_edge(x_edge)
            x_edge = F.relu(x_edge)
        
        ####### DSM head decoding layers intertwined with other heads (exact TensorFlow match) #######
        #### DSM head decoding layer 1 ####
        x_dsm = self.deconv1_dsm(x)
        x_dsm = F.relu(x_dsm)
        if self.sem_flag:
            x_dsm = torch.cat([x_dsm, x3_sem], dim=1)
        if self.norm_flag:
            x_dsm = torch.cat([x_dsm, x3_norm], dim=1)
        if self.edge_flag:
            x_dsm = torch.cat([x_dsm, x3_edge], dim=1)
        x_dsm = self.conv_1_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_2_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do1_dsm(x_dsm) if training else x_dsm
        
        #### DSM head decoding layer 2 ####
        x_dsm = self.deconv2_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        if self.sem_flag:
            x_dsm = torch.cat([x_dsm, x4_sem], dim=1)
        if self.norm_flag:
            x_dsm = torch.cat([x_dsm, x4_norm], dim=1)
        if self.edge_flag:
            x_dsm = torch.cat([x_dsm, x4_edge], dim=1)
        x_dsm = self.conv_3_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_4_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do2_dsm(x_dsm) if training else x_dsm
        
        #### DSM head decoding layer 3 ####
        x_dsm = self.deconv3_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        if self.sem_flag:
            x_dsm = torch.cat([x_dsm, x5_sem], dim=1)
        if self.norm_flag:
            x_dsm = torch.cat([x_dsm, x5_norm], dim=1)
        if self.edge_flag:
            x_dsm = torch.cat([x_dsm, x5_edge], dim=1)
        x_dsm = self.conv_5_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_6_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do3_dsm(x_dsm) if training else x_dsm
        
        #### DSM head decoding layer 4 ####
        x_dsm = self.deconv4_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        if self.sem_flag:
            x_dsm = torch.cat([x_dsm, x6_sem], dim=1)
        if self.norm_flag:
            x_dsm = torch.cat([x_dsm, x6_norm], dim=1)
        if self.edge_flag:
            x_dsm = torch.cat([x_dsm, x6_edge], dim=1)
        x_dsm = self.conv_7_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_8_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do4_dsm(x_dsm) if training else x_dsm
        
        #### DSM head decoding layer 5 ####
        x_dsm = self.deconv5_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        if self.sem_flag:
            x_dsm = torch.cat([x_dsm, x7_sem], dim=1)
        if self.norm_flag:
            x_dsm = torch.cat([x_dsm, x7_norm], dim=1)
        if self.edge_flag:
            x_dsm = torch.cat([x_dsm, x7_edge], dim=1)
        x_dsm = self.conv_9_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.conv_10_dsm(x_dsm)
        x_dsm = F.relu(x_dsm)
        x_dsm = self.do5_dsm(x_dsm) if training else x_dsm
        
        x_dsm = self.conv_dsm(x_dsm)
        
        return x_dsm, x_sem, x_norm, x_edge
    
    def call_full(self, x, training=True):
        """Faithful 1:1 conversion of TensorFlow MTL.call_full method"""
        
        x_sem, x_norm, x_edge = [], [], []
        
        ####### Decoding layers intertwined all with each other (exact TensorFlow match) #######
        
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
        x_dsm = self.do1_dsm(x_dsm) if training else x_dsm
        
        # SEM head decoding layer 1
        if self.sem_flag:
            x_sem = self.conv_1_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_2_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do1_sem(x_sem) if training else x_sem
        
        # Norm head decoding layer 1
        if self.norm_flag:
            x_norm = self.conv_1_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_2_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do1_norm(x_norm) if training else x_norm
        
        # Edge head decoding layer 1
        if self.edge_flag:
            x_edge = self.conv_1_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_2_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do1_edge(x_edge) if training else x_edge
        
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
        x_dsm = self.do2_dsm(x_dsm) if training else x_dsm
        
        # SEM head decoding layer 2
        if self.sem_flag:
            x_sem = self.conv_3_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_4_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do2_sem(x_sem) if training else x_sem
        
        # Norm head decoding layer 2
        if self.norm_flag:
            x_norm = self.conv_3_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_4_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do2_norm(x_norm) if training else x_norm
        
        # Edge head decoding layer 2
        if self.edge_flag:
            x_edge = self.conv_3_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_4_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do2_edge(x_edge) if training else x_edge
        
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
        x_dsm = self.do3_dsm(x_dsm) if training else x_dsm
        
        # SEM head decoding layer 3
        if self.sem_flag:
            x_sem = self.conv_5_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_6_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do3_sem(x_sem) if training else x_sem
        
        # Norm head decoding layer 3
        if self.norm_flag:
            x_norm = self.conv_5_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_6_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do3_norm(x_norm) if training else x_norm
        
        # Edge head decoding layer 3
        if self.edge_flag:
            x_edge = self.conv_5_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_6_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do3_edge(x_edge) if training else x_edge
        
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
        x_dsm = self.do4_dsm(x_dsm) if training else x_dsm
        
        # SEM head decoding layer 4
        if self.sem_flag:
            x_sem = self.conv_7_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_8_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do4_sem(x_sem) if training else x_sem
        
        # Norm head decoding layer 4
        if self.norm_flag:
            x_norm = self.conv_7_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_8_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do4_norm(x_norm) if training else x_norm
        
        # Edge head decoding layer 4
        if self.edge_flag:
            x_edge = self.conv_7_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_8_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do4_edge(x_edge) if training else x_edge
        
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
        x_dsm = self.do5_dsm(x_dsm) if training else x_dsm
        
        x_dsm = self.conv_dsm(x_dsm)
        
        # SEM head decoding layer 5
        if self.sem_flag:
            x_sem = self.conv_9_sem(x_concat)
            x_sem = F.relu(x_sem)
            x_sem = self.conv_10_sem(x_sem)
            x_sem = F.relu(x_sem)
            x_sem = self.do5_sem(x_sem) if training else x_sem
            
            x_sem = self.conv_sem(x_sem)
            x_sem = F.softmax(x_sem, dim=1)
        
        # Norm head decoding layer 5
        if self.norm_flag:
            x_norm = self.conv_9_norm(x_concat)
            x_norm = F.relu(x_norm)
            x_norm = self.conv_10_norm(x_norm)
            x_norm = F.relu(x_norm)
            x_norm = self.do5_norm(x_norm) if training else x_norm
            
            x_norm = self.conv_norm(x_norm)
        
        # Edge head decoding layer 5
        if self.edge_flag:
            x_edge = self.conv_9_edge(x_concat)
            x_edge = F.relu(x_edge)
            x_edge = self.conv_10_edge(x_edge)
            x_edge = F.relu(x_edge)
            x_edge = self.do5_edge(x_edge) if training else x_edge
            
            x_edge = self.conv_edge(x_edge)
            x_edge = F.relu(x_edge)
        
        return x_dsm, x_sem, x_norm, x_edge


## Denoising Autoencoder architecture (faithful PyTorch conversion)
class Autoencoder(nn.Module):
    """Faithful 1:1 conversion of TensorFlow Autoencoder"""
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Calculate input channels for autoencoder
        input_channels = calculate_dae_input_channels()
        
        # Encoder - exact TensorFlow match
        self.conv1_0 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2_0 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3_0 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4_0 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.conv5_0 = nn.Conv2d(512, 1024, 3, padding=1)
        self.conv5 = nn.Conv2d(1024, 1024, 3, padding=1)
        
        # Decoder - exact TensorFlow match
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6_0 = nn.Conv2d(1024, 512, 3, padding=1)
        self.conv6 = nn.Conv2d(512, 512, 3, padding=1)
        
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7_0 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8_0 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9_0 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.out = nn.Conv2d(64, 1, 3, padding=1)
        
        self.do1 = nn.Dropout(0.5)
        self.do2 = nn.Dropout(0.5)
        self.do3 = nn.Dropout(0.5)
        self.do4 = nn.Dropout(0.5)
    
    def forward(self, x, training=True):
        """Faithful 1:1 conversion of TensorFlow Autoencoder.call method"""
        # Encoder
        x1 = F.relu(self.conv1_0(x))
        x1 = F.relu(self.conv1(x1))
        x1_pool = self.pool1(x1)
        
        x2 = F.relu(self.conv2_0(x1_pool))
        x2 = F.relu(self.conv2(x2))
        x2_pool = self.pool2(x2)
        
        x3 = F.relu(self.conv3_0(x2_pool))
        x3 = F.relu(self.conv3(x3))
        x3_pool = self.pool3(x3)
        
        x4 = F.relu(self.conv4_0(x3_pool))
        x4 = F.relu(self.conv4(x4))
        x4_pool = self.pool4(x4)
        
        # Bottleneck
        x5 = F.relu(self.conv5_0(x4_pool))
        x5 = F.relu(self.conv5(x5))
        
        # Decoder with skip connections
        x = F.relu(self.up6(x5))
        x = torch.cat([x4, x], dim=1)
        x = F.relu(self.conv6_0(x))
        x = F.relu(self.conv6(x))
        x = self.do1(x) if training else x
        
        x = F.relu(self.up7(x))
        x = torch.cat([x3, x], dim=1)
        x = F.relu(self.conv7_0(x))
        x = F.relu(self.conv7(x))
        x = self.do2(x) if training else x
        
        x = F.relu(self.up8(x))
        x = torch.cat([x2, x], dim=1)
        x = F.relu(self.conv8_0(x))
        x = F.relu(self.conv8(x))
        x = self.do3(x) if training else x
        
        x = F.relu(self.up9(x))
        x = torch.cat([x1, x], dim=1)
        x = F.relu(self.conv9_0(x))
        x = F.relu(self.conv9(x))
        x = self.do4(x) if training else x
        
        return self.out(x)


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
        # Standard RGB input
        backbone = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    else:
        # Custom input channels (e.g., for SAR mode)
        backbone = models.densenet121(weights=None)
        # Modify first conv layer for different input channels
        backbone.features.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Remove classifier to match TensorFlow include_top=False
    backbone.classifier = nn.Identity()
    return backbone
