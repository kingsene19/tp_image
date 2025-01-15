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
        self.target = VideoSkeleton(filename_tgt, modFrame=20)
        self.source = VideoReader(filename_src)
        print(f"Source: {filename_src}")
        print(f"Target: {filename_tgt}")

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
        for i in range(self.source.getTotalFrames()):
            print(f"Press 'q' to quit. [{i}/{self.source.getTotalFrames()}]", end='\r')
            image_src = self.source.readFrame()
            if i%5 == 0:
                isSke, image_src, ske = self.target.cropAndSke(image_src, ske)
                image_tgt = self.generator.generate(ske) if isSke else image_err
                # rescale images to 512x512
                image_src = cv2.resize(image_src, (512, 512))
                if isSke:
                    ske.draw(image_src)
                image_tgt = cv2.resize(image_tgt, (512, 512))
                    
                image_combined = combineTwoImages(image_src, image_tgt)
                image_combined = cv2.resize(image_combined, (1024, 512))
                cv2.imshow(f"Dance demo - {self.generator.__class__.__name__}", image_combined)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        print("End of demo" + " " * 100)
        
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
           
           
    def save_video(self):
        VIDEO_PATH = f"output/{self.name}-{['Nearest', 'VanillaNN-Ske', 'VanillaNN-Image', 'GAN'][self.typeOfGen-1]}.mp4"

        ske = Skeleton()
        image_err = np.zeros((128, 128, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(VIDEO_PATH, fourcc, 30.0, (256, 256))

        for i in range(self.source.getTotalFrames()):
            print(f"Processing frame {i}/{self.source.getTotalFrames()}", end='\r')
            image_src = self.source.readFrame()
            isSke, image_src, ske = self.target.cropAndSke(image_src, ske)
            image_tgt = self.generator.generate(ske) if isSke else image_err
            # resize to 256x256
            image_tgt = cv2.resize(image_tgt, (256, 256))
            out.write(image_tgt)
        out.release()
        cv2.destroyAllWindows()
        print(f"Video saved to {VIDEO_PATH}")
        return VIDEO_PATH
        
        
        
if __name__ == '__main__':
    # select device
    if sys.platform == 'darwin':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    while True:
        print("======== Select the type of generation =======")
        GEN_TYPE = input("Which generation type would you like?\n[1].GenNearest\n[2].GenSkeToImage\n[3].GenImageToImage\n[4].GenGAN\n(5).Quit\nAnswer:")
        if GEN_TYPE in ['5', 'q', 'Q']:
            print("Exiting...")
            break
        if GEN_TYPE not in ['1', '2', '3', '4']:
            print("Invalid input. Please try again.")
            continue
        
        GEN_NAMES = ['Nearest', 'VanillaNN-Ske', 'VanillaNN-Image', 'GAN']
        GEN_TYPE = int(GEN_TYPE)
        ddemo = DanceDemo("data/taichi2.mp4", "data/taichi1.mp4", GEN_TYPE, False)
        print(f"======== Dance demo [{GEN_NAMES[int(GEN_TYPE)-1]}] =======")
        ddemo.draw()
