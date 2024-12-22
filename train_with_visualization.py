import os

from keras.callbacks import TensorBoard

from models.auto_encoder import AutoEncoder
from utils import ImageLoader, PredictCallback

DATA_DIR = os.path.join("screw")
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
VALID_DATA_DIR = os.path.join(DATA_DIR, "valid")
LOGS_DIR = os.path.join("logs", "auto_encoder_visualization")
DEBUG_DIR = os.path.join("debugs")
MODEL_FILEPATH = os.path.join("results", "autoencoder_experiment_01.h5")

model = AutoEncoder()
image_size = model.input.shape[1:3]

model.summary()

train_images = ImageLoader(TRAIN_DATA_DIR, tuple(image_size)).load()
valid_images = ImageLoader(VALID_DATA_DIR, tuple(image_size)).load()

targets = train_images[:9] + valid_images[:9]
predict_callback = PredictCallback(targets, DEBUG_DIR, period=5, verbose=1)
tensorboard_callback = TensorBoard(log_dir=LOGS_DIR)

model.fit(
    train_images,
    train_images,
    validation_data=(valid_images, valid_images),
    batch_size=125,
    epochs=500,
    callbacks=[tensorboard_callback, predict_callback],
)

model.save(MODEL_FILEPATH)
