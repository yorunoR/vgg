import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def VGG16():
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        input_layer = Input(shape=(224, 224, 3))

        conv11 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
        conv12 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv11)
        pool1 = MaxPooling2D()(conv12)

        conv21 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
        conv22 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv21)
        pool2 = MaxPooling2D()(conv22)

        conv31 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool2)
        conv32 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv31)
        conv33 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv32)
        pool3 = MaxPooling2D()(conv33)

        conv41 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
        conv42 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv41)
        conv43 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv42)
        pool4 = MaxPooling2D()(conv43)

        conv51 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool4)
        conv52 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv51)
        conv53 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv52)
        pool5 = MaxPooling2D()(conv53)

        flat = Flatten()(pool5)
        fc1 = Dense(4096, activation="relu")(flat)
        dropout1 = Dropout(0.5)(fc1)
        fc2 = Dense(4096, activation="relu")(dropout1)
        dropout2 = Dropout(0.5)(fc2)
        output_layer = Dense(3, activation="softmax")(dropout2)

        model = Model(inputs=input_layer, outputs=output_layer)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=1e-2,  # 初期学習率
            decay_steps=1000,  # 減衰ステップ数
            decay_rate=0.96,  # 減衰率
            staircase=True,  # ステップ単位の減衰（Trueの場合）
        )

        sgd = SGD(learning_rate=lr_schedule, momentum=0.9)
        model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

    return model
