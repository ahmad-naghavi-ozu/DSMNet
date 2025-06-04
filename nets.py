# MAHDI ELHOUSNI, WPI 2020 
# Altered by Ahmad Naghavi, OzU 2024
# Converted to PyTorch by Ahmad Naghavi, OzU 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from config import *

class MTL(nn.Module):
    """Multi-task Learning (MTL) architecture using PyTorch"""
    
    def __init__(self, backbone_net, sem_flag=True, norm_flag=True, edge_flag=True):
        super(MTL, self).__init__()
        
        # Determine the initial convolution layer to turn the RGB+SAR 4-channel input to a 3-channel tensor.
        # This is mandatory for the sake of DenseNet121 input as the number of channels shall be three.
        if sar_mode:
            self.conv_sar = nn.Conv2d(4, 3, kernel_size=3, padding=1)
            self.do_sar = nn.Dropout(0.5)
        
        # Store backbone and flags
        self.encoder = backbone_net
        self.sem_flag = sem_flag
        self.norm_flag = norm_flag
        self.edge_flag = edge_flag
        
        self.bn0 = nn.BatchNorm2d(1024)  # Assuming encoder output has 1024 channels
        
        # Establish decoding layers for SEM (Semantic Segmentation)
        if self.sem_flag:
            self.deconv1_sem = nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1)
            self.deconv2_sem = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
            self.deconv3_sem = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
            self.deconv4_sem = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
            self.deconv5_sem = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            
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
            
            self.conv_edge = nn.Conv2d(32, 1, 3, stride=1, padding=1)
            
            self.do1_edge = nn.Dropout(0.5)
            self.do2_edge = nn.Dropout(0.5)
            self.do3_edge = nn.Dropout(0.5)
            self.do4_edge = nn.Dropout(0.5)
            self.do5_edge = nn.Dropout(0.5)
        
        # Establish decoding layers for DSM (Height estimation)
        self.deconv1_dsm = nn.ConvTranspose2d(1024, 1024, 3, stride=2, padding=1, output_padding=1)
        self.deconv2_dsm = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv3_dsm = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv4_dsm = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv5_dsm = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        
        self.conv_1_dsm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.conv_2_dsm = nn.Conv2d(1024, 1024, 3, stride=1, padding=1)
        self.conv_3_dsm = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_dsm = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_dsm = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_6_dsm = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_7_dsm = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_8_dsm = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_9_dsm = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv_10_dsm = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        
        self.conv_dsm = nn.Conv2d(32, 1, 3, stride=1, padding=1)
        
        self.do1_dsm = nn.Dropout(0.5)
        self.do2_dsm = nn.Dropout(0.5)
        self.do3_dsm = nn.Dropout(0.5)
        self.do4_dsm = nn.Dropout(0.5)
        self.do5_dsm = nn.Dropout(0.5)

    def forward(self, x, head_mode='full', training=True):
        """Forward pass of the MTL network
        
        Args:
            x: Input tensor
            head_mode: Either 'full' for all heads or 'dsm' for DSM head only
            training: Whether in training mode
        """
        # Handle SAR mode if enabled
        if sar_mode:
            x = self.conv_sar(x)
            x = self.do_sar(x)
        
        # Pass through encoder (backbone)
        features = self.encoder.features(x)
        features = self.bn0(features)
        
        # DSM head (always computed)
        dsm_out = self._decode_dsm(features, training)
        
        # Optional heads based on flags and head_mode
        sem_out = None
        norm_out = None  
        edge_out = None
        
        if head_mode == 'full':
            if self.sem_flag:
                sem_out = self._decode_sem(features, training)
            if self.norm_flag:
                norm_out = self._decode_norm(features, training)
            if self.edge_flag:
                edge_out = self._decode_edge(features, training)
        
        return dsm_out, sem_out, norm_out, edge_out
    
    def _decode_dsm(self, features, training=True):
        """Decode DSM (height) predictions"""
        x = F.relu(self.conv_1_dsm(features))
        x = self.do1_dsm(x) if training else x
        x = F.relu(self.conv_2_dsm(x))
        x = self.do1_dsm(x) if training else x
        x = self.deconv1_dsm(x)
        
        x = F.relu(self.conv_3_dsm(x))
        x = self.do2_dsm(x) if training else x
        x = F.relu(self.conv_4_dsm(x))
        x = self.do2_dsm(x) if training else x
        x = self.deconv2_dsm(x)
        
        x = F.relu(self.conv_5_dsm(x))
        x = self.do3_dsm(x) if training else x
        x = F.relu(self.conv_6_dsm(x))
        x = self.do3_dsm(x) if training else x
        x = self.deconv3_dsm(x)
        
        x = F.relu(self.conv_7_dsm(x))
        x = self.do4_dsm(x) if training else x
        x = F.relu(self.conv_8_dsm(x))
        x = self.do4_dsm(x) if training else x
        x = self.deconv4_dsm(x)
        
        x = F.relu(self.conv_9_dsm(x))
        x = self.do5_dsm(x) if training else x
        x = F.relu(self.conv_10_dsm(x))
        x = self.do5_dsm(x) if training else x
        x = self.deconv5_dsm(x)
        
        return self.conv_dsm(x)
    
    def _decode_sem(self, features, training=True):
        """Decode semantic segmentation predictions"""
        x = F.relu(self.conv_1_sem(features))
        x = self.do1_sem(x) if training else x
        x = F.relu(self.conv_2_sem(x))
        x = self.do1_sem(x) if training else x
        x = self.deconv1_sem(x)
        
        x = F.relu(self.conv_3_sem(x))
        x = self.do2_sem(x) if training else x
        x = F.relu(self.conv_4_sem(x))
        x = self.do2_sem(x) if training else x
        x = self.deconv2_sem(x)
        
        x = F.relu(self.conv_5_sem(x))
        x = self.do3_sem(x) if training else x
        x = F.relu(self.conv_6_sem(x))
        x = self.do3_sem(x) if training else x
        x = self.deconv3_sem(x)
        
        x = F.relu(self.conv_7_sem(x))
        x = self.do4_sem(x) if training else x
        x = F.relu(self.conv_8_sem(x))
        x = self.do4_sem(x) if training else x
        x = self.deconv4_sem(x)
        
        x = F.relu(self.conv_9_sem(x))
        x = self.do5_sem(x) if training else x
        x = F.relu(self.conv_10_sem(x))
        x = self.do5_sem(x) if training else x
        x = self.deconv5_sem(x)
        
        x = self.conv_sem(x)
        return F.softmax(x, dim=1)  # Apply softmax for semantic segmentation
    
    def _decode_norm(self, features, training=True):
        """Decode surface normal predictions"""
        x = F.relu(self.conv_1_norm(features))
        x = self.do1_norm(x) if training else x
        x = F.relu(self.conv_2_norm(x))
        x = self.do1_norm(x) if training else x
        x = self.deconv1_norm(x)
        
        x = F.relu(self.conv_3_norm(x))
        x = self.do2_norm(x) if training else x
        x = F.relu(self.conv_4_norm(x))
        x = self.do2_norm(x) if training else x
        x = self.deconv2_norm(x)
        
        x = F.relu(self.conv_5_norm(x))
        x = self.do3_norm(x) if training else x
        x = F.relu(self.conv_6_norm(x))
        x = self.do3_norm(x) if training else x
        x = self.deconv3_norm(x)
        
        x = F.relu(self.conv_7_norm(x))
        x = self.do4_norm(x) if training else x
        x = F.relu(self.conv_8_norm(x))
        x = self.do4_norm(x) if training else x
        x = self.deconv4_norm(x)
        
        x = F.relu(self.conv_9_norm(x))
        x = self.do5_norm(x) if training else x
        x = F.relu(self.conv_10_norm(x))
        x = self.do5_norm(x) if training else x
        x = self.deconv5_norm(x)
        
        return torch.tanh(self.conv_norm(x))  # Tanh for normalized surface normals
    
    def _decode_edge(self, features, training=True):
        """Decode edge map predictions"""
        x = F.relu(self.conv_1_edge(features))
        x = self.do1_edge(x) if training else x
        x = F.relu(self.conv_2_edge(x))
        x = self.do1_edge(x) if training else x
        x = self.deconv1_edge(x)
        
        x = F.relu(self.conv_3_edge(x))
        x = self.do2_edge(x) if training else x
        x = F.relu(self.conv_4_edge(x))
        x = self.do2_edge(x) if training else x
        x = self.deconv2_edge(x)
        
        x = F.relu(self.conv_5_edge(x))
        x = self.do3_edge(x) if training else x
        x = F.relu(self.conv_6_edge(x))
        x = self.do3_edge(x) if training else x
        x = self.deconv3_edge(x)
        
        x = F.relu(self.conv_7_edge(x))
        x = self.do4_edge(x) if training else x
        x = F.relu(self.conv_8_edge(x))
        x = self.do4_edge(x) if training else x
        x = self.deconv4_edge(x)
        
        x = F.relu(self.conv_9_edge(x))
        x = self.do5_edge(x) if training else x
        x = F.relu(self.conv_10_edge(x))
        x = self.do5_edge(x) if training else x
        x = self.deconv5_edge(x)
        
        return torch.sigmoid(self.conv_edge(x))  # Sigmoid for edge maps


class UNet(nn.Module):
    """U-Net architecture for denoising autoencoder (DAE)"""
    
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (downsampling)
        self.enc_conv1 = self._conv_block(in_channels, 64)
        self.enc_conv2 = self._conv_block(64, 128)
        self.enc_conv3 = self._conv_block(128, 256)
        self.enc_conv4 = self._conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._conv_block(512, 1024)
        
        # Decoder (upsampling)
        self.dec_conv4 = self._conv_block(1024 + 512, 512)
        self.dec_conv3 = self._conv_block(512 + 256, 256)
        self.dec_conv2 = self._conv_block(256 + 128, 128)
        self.dec_conv1 = self._conv_block(128 + 64, 64)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Max pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def _conv_block(self, in_channels, out_channels):
        """Create a convolutional block with two conv layers"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(self.pool(enc1))
        enc3 = self.enc_conv3(self.pool(enc2))
        enc4 = self.enc_conv4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec4 = self.dec_conv4(torch.cat([self.upsample(bottleneck), enc4], dim=1))
        dec3 = self.dec_conv3(torch.cat([self.upsample(dec4), enc3], dim=1))
        dec2 = self.dec_conv2(torch.cat([self.upsample(dec3), enc2], dim=1))
        dec1 = self.dec_conv1(torch.cat([self.upsample(dec2), enc1], dim=1))
        
        return self.final_conv(dec1)


def create_densenet_backbone(input_channels=3):
    """Create DenseNet121 backbone for MTL network"""
    # Load pretrained DenseNet121
    densenet = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    
    # Modify first conv layer if input channels != 3
    if input_channels != 3:
        original_conv = densenet.features.conv0
        densenet.features.conv0 = nn.Conv2d(
            input_channels, 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias
        )
        
        # Initialize new conv layer weights
        with torch.no_grad():
            if input_channels > 3:
                # Duplicate RGB weights and add zeros for extra channels
                densenet.features.conv0.weight[:, :3] = original_conv.weight
                densenet.features.conv0.weight[:, 3:].zero_()
            else:
                # Average RGB weights for fewer channels
                densenet.features.conv0.weight = original_conv.weight[:, :input_channels].mean(dim=1, keepdim=True)
    
    # Remove the classifier (we only need features)
    return densenet


def setup_model_for_multi_gpu(model, device_ids=None):
    """Setup model for multi-GPU training using DataParallel or DistributedDataParallel"""
    if device_ids is None:
        device_ids = GPU_IDS
    
    if USE_MULTI_GPU and len(device_ids) > 1 and torch.cuda.device_count() > 1:
        if dist.is_initialized():
            # Use DistributedDataParallel for better performance
            model = nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[device_ids[dist.get_rank()]],
                output_device=device_ids[dist.get_rank()]
            )
        else:
            # Use DataParallel as fallback
            model = nn.DataParallel(model, device_ids=device_ids)
        print(f"Model wrapped for multi-GPU training on devices: {device_ids}")
    
    return model
