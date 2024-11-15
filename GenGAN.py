import numpy as np
import cv2
import os
import pickle
import sys
import math
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch import Tensor
from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import * 

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)

    
class GenGAN():
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
    """
    def __init__(self, videoSke, name, loadFromFile=False):
        self.netG = GenNNSkeImToImage()
        self.netD = Discriminator()
        self.real_labels = 1.
        self.fake_labels = 0.
        self.filename = f"models/DanceGenGAN-{name}.pth"
        tgt_transform = transforms.Compose(
                            [transforms.Resize((128, 128)),
                            transforms.CenterCrop(128),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        src_transform = SkeToImageTransform(128)
        self.dataset = VideoSkeletonDataset(videoSke, ske_reduced=True, target_transform=tgt_transform, source_transform=src_transform)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=32, shuffle=True)
        if loadFromFile and os.path.isfile(self.filename):
            print("GenGAN: Load=", self.filename, "   Current Working Directory=", os.getcwd())
            self.netG = torch.load(self.filename, map_location=torch.device('cpu'))


    def train(self, n_epochs=20):
        optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))  
        optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        gan_loss_fn = nn.BCELoss()
        l1_loss_fn = nn.L1Loss()
        self.netG.train()
        self.netD.train()
        try:
            for epoch in range(n_epochs):
                for i, (ske, images) in enumerate(self.dataloader):
                    batch_size = images.shape[0]
                    fake_images = self.netG(ske)
                    real = torch.full((batch_size,), self.real_labels)
                    fake = torch.full((batch_size,), self.fake_labels)

                    # Train Discriminator
                    optimizerD.zero_grad()
                    
                    output = self.netD(images)
                    errD_real = gan_loss_fn(output, real)

                    output = self.netD(fake_images.detach())
                    errD_fake = gan_loss_fn(output, fake)
                    
                    errD = errD_real + errD_fake
                    
                    errD.backward()
                    optimizerD.step()
                    
                    # Train Generator
                    optimizerG.zero_grad()
                    output = self.netD(fake_images)
                    errG = 100 * l1_loss_fn(fake_images, images) + 1.0 * gan_loss_fn(output, real)
                    errG.backward()
                    optimizerG.step() 

                print(f'Epoch [{epoch + 1}/{n_epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}', end='\r')
                torch.save(self.netG, self.filename)
                print(f"Model was saved at {epoch+1}")
        except KeyboardInterrupt:
            print("GenGAN: Training interrupted")
            
        torch.save(self.netG, self.filename)
        print(f"Model was saved at {epoch+1}")
            

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
    print("GenGAN: Current Working Directory=", os.getcwd())
    print("GenGAN: Filename=", filename)

    targetVideoSke = VideoSkeleton(filename)

    gen = GenGAN(targetVideoSke, filename, False)
    gen.train(10)