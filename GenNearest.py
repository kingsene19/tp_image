import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton, combineTwoImages
from VideoReader import VideoReader
from Skeleton import Skeleton

class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Nearest neighbor method: it selects the image in videoSke that has the skeleton closest to the input skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt

    def generate(self, ske):           
        """ generator of image from skeleton """
        min_distance = float('inf')
        index = -1
        for i in range(self.videoSkeletonTarget.skeCount()):
            distance = ske.distance(self.videoSkeletonTarget.ske[i])
            if distance < min_distance:
                min_distance = distance
                index = i

        if index != -1:
            image = self.videoSkeletonTarget.readImage(index)
            return image
        
        empty = np.ones((128, 128, 3), dtype=np.uint8) * 255
        return empty
