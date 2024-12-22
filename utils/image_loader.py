import glob
import os

import cv2
import numpy as np


class ImageLoader:
    def __init__(self, dir_path, image_size):
        self.__dir_path = dir_path
        self.__image_size = image_size

    def load(self):
        images = []
        files = glob.glob(os.path.join(self.__dir_path, "*.png"))
        files.sort()
        for file_path in files:
            img = cv2.imread(file_path)
            img = cv2.resize(img, self.__image_size)
            img = img / 255
            images.append(img)
        images = np.array(images)
        return images
