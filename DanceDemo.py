import numpy as np
import cv2
import os
import pickle
import sys

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
    def __init__(self, filename_src, filename_tgt, typeOfGen=2):
        self.filename = filename_tgt
        self.target = VideoSkeleton(self.filename)
        self.typeOfGen = typeOfGen
        self.source = VideoReader(filename_src)
        if typeOfGen==1:           # Nearest
            print("Generator: GenNeirest")
            self.generator = GenNeirest(self.target)
        elif typeOfGen==2:         # VanillaNN
            print("Generator: GenSimpleNN")
            if os.path.exists("data/DanceGenVanillaFromSke"+self.filename.split("/")[1].split(".")[0]+"1"+".pth"):
                self.generator = GenVanillaNN( self.target, self.filename, loadFromFile=True, optSkeOrImage=1)
            else:
                self.generator = GenVanillaNN( self.target, self.filename, loadFromFile=False, optSkeOrImage=1)
                self.generator.train()
        elif typeOfGen==3:         # VanillaNN
            print("Generator: GenSimpleNN")
            if os.path.exists("data/DanceGenVanillaFromSke"+self.filename.split("/")[1].split(".")[0]+"2"+".pth"):
                self.generator = GenVanillaNN( self.target, self.filename, loadFromFile=True, optSkeOrImage=2)
            else:
                self.generator = GenVanillaNN( self.target,self.filename, loadFromFile=False, optSkeOrImage=2)
                self.generator.train()
        elif typeOfGen==4:         # GAN
            print("Generator: GenSimpleNN")
            if os.path.exists("data/DanceGenGAN"+self.filename.split("/")[1].split(".")[0]+".pth"):
                self.generator = GenGAN( self.target,self.filename, loadFromFile=True)
            else:
                self.generator = GenGAN( self.target,self.filename, loadFromFile=False)
                self.generator.train()
        else:
            print("DanceDemo: typeOfGen error!!!")


    def draw(self):
        ske = Skeleton()
        image_err = np.zeros((128, 128, 3), dtype=np.uint8)
        image_err[:, :] = (0, 0, 255)  # (B, G, R)
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



if __name__ == '__main__':
    GEN_TYPE = 4
    ddemo = DanceDemo("data/taichi2_full.mp4", "data/taichi1.mp4", GEN_TYPE)
    ddemo.draw()
