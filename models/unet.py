import tensorflow as tf
from keras.layers import Concatenate, Conv2D, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def Unet(class_num):
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        input_layer = Input(shape=(128, 128, 3))

        conv11 = Conv2D(64, (3, 3), activation="relu", padding="same")(input_layer)
        conv12 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv11)
        pool1 = MaxPooling2D()(conv12)

        conv21 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool1)
        conv22 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv21)
        pool2 = MaxPooling2D()(conv22)

        conv31 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool2)
        conv32 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv31)

        up1 = UpSampling2D()(conv32)
        concat1 = Concatenate()([up1, conv22])
        conv41 = Conv2D(128, (3, 3), activation="relu", padding="same")(concat1)
        conv42 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv41)

        up2 = UpSampling2D()(conv42)
        concat2 = Concatenate()([up2, conv12])
        conv51 = Conv2D(128, (3, 3), activation="relu", padding="same")(concat2)
        conv52 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv51)

        output_layer = Conv2D(3, (1, 1), activation="relu", padding="same")(conv52)

        model = Model(inputs=input_layer, outputs=output_layer)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=1e-2,  # 初期学習率
            decay_steps=1000,  # 減衰ステップ数
            decay_rate=0.96,  # 減衰率
            staircase=True,  # ステップ単位の減衰（Trueの場合）
        )

        adam = Adam(learning_rate=lr_schedule)
        model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["acc"])

        return model
