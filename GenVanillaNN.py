import numpy as np
import cv2
import os
import pickle
import sys
import math

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.io import read_image

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms

#from tensorboardX import SummaryWriter

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton

torch.set_default_dtype(torch.float32)

class SkeToImageTransform:
    def __init__(self, image_size):
        self.imsize = image_size

    def __call__(self, ske):
        image = np.zeros((self.imsize, self.imsize, 3), dtype=np.uint8)
        Skeleton.draw_reduced(ske.reduce(), image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float()
        return image

class VideoSkeletonDataset(Dataset):
    def __init__(self, videoSke, ske_reduced, source_transform=None, target_transform=None):
        """ videoSkeleton dataset: 
                videoske(VideoSkeleton): video skeleton that associate a video and a skeleton for each frame
                ske_reduced(bool): use reduced skeleton (13 joints x 2 dim=26) or not (33 joints x 3 dim = 99)
        """
        self.videoSke = videoSke
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.ske_reduced = ske_reduced
        print("VideoSkeletonDataset: ",
              "ske_reduced=", ske_reduced, "=(", Skeleton.reduced_dim, " or ",Skeleton.full_dim,")" )


    def __len__(self):
        return self.videoSke.skeCount()


    def __getitem__(self, idx):
        # prepreocess skeleton (input)
        reduced = True
        ske = self.videoSke.ske[idx]
        ske = self.preprocessSkeleton(ske)
        # prepreocess image (output)
        image = Image.open(self.videoSke.imagePath(idx))
        if self.target_transform:
            image = self.target_transform(image)
        return ske, image

    
    def preprocessSkeleton(self, ske):
        if self.source_transform:
            ske = self.source_transform(ske)
        else:
            ske = torch.from_numpy( ske.__array__(reduced=self.ske_reduced).flatten() )
            ske = ske.to(torch.float32)
            ske = ske.reshape(ske.shape[0],1,1)
        return ske


    def tensor2image(self, normalized_image):
        numpy_image = normalized_image.detach().numpy()
        numpy_image = np.transpose(numpy_image, (1, 2, 0))
        numpy_image = cv2.cvtColor(np.array(numpy_image), cv2.COLOR_RGB2BGR)
        denormalized_image = numpy_image * np.array([0.5, 0.5, 0.5]) + np.array([0.5, 0.5, 0.5])
        denormalized_image = (denormalized_image * 255).astype(np.uint8)
        return denormalized_image


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Transform input features
        proj_query = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width*height)
        attention = self.softmax(torch.bmm(proj_query, proj_key))
        
        proj_value = self.value(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels)
        )
        
    def forward(self, x):
        return x + self.block(x)

class GenNNSkeToImage(nn.Module):
    def __init__(self, latent_dim=26):
        super(GenNNSkeToImage, self).__init__()
        
        # Initial projection
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Upsampling layers
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Add self-attention after 16x16 resolution
        self.attention = SelfAttention(128)
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual blocks for better feature processing
        self.res_blocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        
        # Final upsampling and output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Initial projection and upsampling
        x = self.initial(x)  # 4x4
        x = self.up1(x)      # 8x8
        x = self.up2(x)      # 16x16
        
        # Apply self-attention
        x = self.attention(x)
        
        # Continue upsampling
        x = self.up3(x)      # 32x32
        x = self.up4(x)      # 64x64
        
        # Apply residual blocks
        x = self.res_blocks(x)
        
        # Final upsampling and output
        x = self.final(x)    # 128x128
        
        return x
    

class GenNNSkeImToImage(nn.Module):
    def __init__(self):
        super(GenNNSkeImToImage, self).__init__()
        
        # Encoder
        self.e1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 128x128
        
        self.e2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 64x64
        
        self.e3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 32x32
        
        self.attention = SelfAttention(256)
        
        self.e4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 16x16
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512)
        )
        
        # Decoder
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 32x32
        
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # 512 = 256 + 256 (skip connection)
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 64x64
        
        self.d3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # 256 = 128 + 128 (skip connection)
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )  # 128x128
        
        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 128 = 64 + 64 (skip connection)
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e3_att = self.attention(e3)
        e4 = self.e4(e3_att)
        
        # Bottleneck
        bottleneck = self.bottleneck(e4)
        
        # Decoder with skip connections
        d1 = self.d1(bottleneck)
        d1_cat = torch.cat([d1, e3], dim=1)
        
        d2 = self.d2(d1_cat)
        d2_cat = torch.cat([d2, e2], dim=1)
        
        d3 = self.d3(d2_cat)
        d3_cat = torch.cat([d3, e1], dim=1)
        
        output = self.final(d3_cat)
        
        return output

class GenVanillaNN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, loadFromFile=False, optSkeOrImage=1):
        input_size=128
        if optSkeOrImage == 1:
            src_transform = None
            filename = "models/DanceGenVanillaFromSke.pth"
            self.netG = GenNNSkeToImage()
        else:
            src_transform = SkeToImageTransform(input_size)
            filename = "models/DanceGenVanillaFromSkeim.pth"
            self.netG = GenNNSkeImToImage()
        
        directory = os.path.dirname(__file__)
        self.filename = os.path.join(directory, filename)
        tgt_transform = transforms.Compose([
                            transforms.Resize(input_size),
                            transforms.CenterCrop(input_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenVanillaNN: Load=", self.filename)
            print("GenVanillaNN: Current Working Directory: ", os.getcwd())
            self.netG = torch.load(self.filename, map_location=torch.device('cpu'))


    def train(self, n_epochs=20):
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.netG.train()
        for epoch in range(n_epochs):
            for i, (ske, target_image) in enumerate(self.dataloader):
                optimizer.zero_grad()
                output_image = self.netG(ske)
                loss = loss_fn(output_image, target_image)
                loss.backward()
                optimizer.step()  
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}", end='\r')
            if (epoch+1) % 20 == 0:
                torch.save(self.netG, self.filename)
                print(f"\nModel was saved at {epoch+1}")
        torch.save(self.netG, self.filename)
        print("Final model was saved")


    def generate(self, ske):
        """ generator of image from skeleton """
        self.netG.eval()
        with torch.no_grad():
            ske_t = self.dataset.preprocessSkeleton(ske)
            ske_t_batch = ske_t.unsqueeze(0)
            normalized_output = self.netG(ske_t_batch)
            res = self.dataset.tensor2image(normalized_output[0])  
            return res


if __name__ == '__main__':
    force = False


    if len(sys.argv) > 1:
        filename = sys.argv[1]
        if len(sys.argv) > 2:
            force = sys.argv[2].lower() == "true"
    else:
        filename = "data/taichi1.mp4"
    print("GenVanillaNN: Current Working Directory=", os.getcwd())
    print("GenVanillaNN: Filename=", filename)
    print("GenVanillaNN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=1)
    gen.train(50)
    gen = GenVanillaNN(targetVideoSke, loadFromFile=False, optSkeOrImage=2)
    gen.train(50)
