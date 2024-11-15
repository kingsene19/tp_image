import numpy as np
import cv2
import os
import pickle
import sys
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use('Agg')

from VideoSkeleton import VideoSkeleton
from VideoSkeleton import combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenNearest import GenNeirest
from GenVanillaNN import *
from GenGAN import *

class DanceDemo:
    """ class that run a demo of the dance.
        The animation/posture from self.source is applied to character define self.target using self.gen
    """
    def __init__(self, filename_src, filename_tgt, typeOfGen=2, train=False, epochs=20):
        self.typeOfGen = typeOfGen
        directory = os.path.dirname(__file__)
        
        filename_src = os.path.join(directory, filename_src)
        filename_tgt = os.path.join(directory, filename_tgt)
        self.name = os.path.basename(filename_tgt).split('.')[0]
        self.target = VideoSkeleton(filename_tgt)
        self.source = VideoReader(filename_src)
        print('---')
        print(f"Source: {filename_src}")
        print(f"Target: {filename_tgt}")
        print(f"Name: {self.name}")
        print('---')
        
        if typeOfGen==1:           # Nearest
            print("Generator: GenNeirest")
            self.generator = GenNeirest(self.target)
        elif typeOfGen==2:         # VanillaNN
            print("Generator: GenSkeToImage")
            if os.path.exists(f"models/DanceGenVanillaFromSke-{self.name}.pth") and not train:
                self.generator = GenVanillaNN( self.target, self.name, loadFromFile=True, optSkeOrImage=1)
            else:
                self.generator = GenVanillaNN( self.target, self.name, loadFromFile=False, optSkeOrImage=1)
                self.generator.train(epochs)
        elif typeOfGen==3:         # VanillaNN
            print("Generator: GenImageToImage")
            if os.path.exists(f"models/DanceGenVanillaFromSkeim-{self.name}.pth") and not train:
                self.generator = GenVanillaNN( self.target, self.name, loadFromFile=True, optSkeOrImage=2)
            else:
                self.generator = GenVanillaNN( self.target,self.name, loadFromFile=False, optSkeOrImage=2)
                self.generator.train(epochs)
        elif typeOfGen==4:         # GAN
            print("Generator: GAN")
            if os.path.exists(f"models/DanceGenGAN-{self.name}.pth") and not train:
                self.generator = GenGAN( self.target,self.name, loadFromFile=True)
            else:
                self.generator = GenGAN( self.target,self.name, loadFromFile=False)
                self.generator.train(epochs)
        else:
            print("DanceDemo: typeOfGen error!!!")


    def draw(self):
        ske = Skeleton()
        image_err = np.zeros((128, 128, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)  # (B, G, R)
        print("Press 'q' to quit")
        print(self.source.getTotalFrames())
        for i in range(self.source.getTotalFrames()):
            image_src = self.source.readFrame()
            if i%5 == 0:
                isSke, image_src, ske = self.target.cropAndSke(image_src, ske)
                if isSke:
                    ske.draw(image_src)
                    image_tgt = self.generator.generate(ske)
                else:
                    image_tgt = image_err
                image_combined = combineTwoImages(image_src, image_tgt)
                image_combined = cv2.resize(image_combined, (512, 256))
                cv2.imshow('Image', image_combined)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                if key & 0xFF == ord('n'):
                    self.source.readNFrames( 100 )
        cv2.destroyAllWindows()
        
    def plot(self, n=10):
        col = 5
        row = (n * 2) // col
        
        print(f"Plotting {n} frames")
        
        ske = Skeleton()
        image_err = np.zeros((128, 128, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)
        
        plt.figure(figsize=(col * 2, row * 2))
        for i in range(n):
            index = max(0, min(self.source.getTotalFrames() // n * i, self.source.getTotalFrames()-1))
            self.source.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            image_src = self.source.readFrame()  
            isSke, image_src, ske = self.target.cropAndSke(image_src, ske)
            duration = -1
            if isSke:
                ske.draw(image_src)
                st = time.time()
                image_tgt = self.generator.generate(ske)
                duration = time.time() - st
            else:
                image_tgt = image_err
            
            r = i // col
            c = i % col 
            plot_index = col * r * 2 + c + 1
            plt.subplot(row, col, plot_index)
            plt.imshow(cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"Frame {index}")
            
            plot_index = col * (r * 2 + 1) + c + 1
            plt.subplot(row, col, plot_index)
            plt.imshow(cv2.cvtColor(image_tgt, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title(f"Generated {index} \n({duration:.2f} s)")
            print(f"Generated {index} ({duration:.2f} s)")

        plt.tight_layout()
        typeGen = "Nearest" if self.typeOfGen==1 else "VanillaNN-Ske" if self.typeOfGen==2 else "VanillaNN-Image" if self.typeOfGen==3 else "GAN"
        directory = os.path.dirname(__file__)
        output_dir = os.path.join(directory, "output")
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{self.name}-{typeGen}.png"))
        plt.show()
            
if __name__ == '__main__':
    GEN_TYPE = eval(input("Which generation type would you like?\n1.GenNearest\n2.GenSkeToImage\n3.GenImageToImage\n4.GenGAN\nAnswer:"))
    TRAIN = False
    EPOCHS = 20
    if GEN_TYPE != 1:
        TRAIN = bool(eval(input("Do you want to train the network or load weights?\n0.False\n1.True\nAnswer:")))
        if TRAIN:
            EPOCHS = eval(input("How many epochs of training do you want?\nAnswer:"))
    ddemo = DanceDemo("data/taichi2_full.mp4", "data/taichi1.mp4", GEN_TYPE, TRAIN, epochs=EPOCHS)
    ddemo.draw()
