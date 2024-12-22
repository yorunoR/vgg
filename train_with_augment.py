import glob
import os

import cv2
import numpy as np
from keras.callbacks import TensorBoard
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from models.vgg16 import VGG16

DATA_DIR = "data"
LOGS_DIR = os.path.join("logs", "vgg16_augmented")

classes = ["person", "cat", "dog"]

model = VGG16()
input_H, input_W = model.input.shape[1:3]

images = {}
labels = {}

for mode in ["train", "valid"]:
    images[mode] = []
    labels[mode] = []
    for class_index, class_name in enumerate(classes):
        files = glob.glob(os.path.join(DATA_DIR, mode, class_name, "*.jpg"))
        for file_path in files:
            img = cv2.imread(file_path)
            img = cv2.resize(img, (input_W, input_H))
            img = img / 255
            images[mode].append(img)
            labels[mode].append(to_categorical(class_index, num_classes=len(classes)))
    images[mode] = np.array(images[mode])
    labels[mode] = np.array(labels[mode])

data_gen = ImageDataGenerator(rotation_range=30, horizontal_flip=True, vertical_flip=True)
tensorboard_callback = TensorBoard(log_dir=LOGS_DIR)
model.fit(
    data_gen.flow(images["train"], labels["train"], batch_size=128),
    validation_data=(images["valid"], labels["valid"]),
    steps_per_epoch=len(images["train"]) // 128,
    epochs=200,
    callbacks=[tensorboard_callback],
)

model.save(os.path.join("results", "vgg16_200epock_lr0.01_3000images_augmented.h5"))
