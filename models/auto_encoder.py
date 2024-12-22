import tensorflow as tf
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay


def AutoEncoder():
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        input_layer = Input(shape=(128, 128, 3))

        conv11 = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
        conv12 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv11)
        pool1 = MaxPooling2D()(conv12)

        conv21 = Conv2D(16, (3, 3), activation="relu", padding="same")(pool1)
        conv22 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv21)
        pool2 = MaxPooling2D()(conv22)

        conv31 = Conv2D(8, (3, 3), activation="relu", padding="same")(pool2)
        conv32 = Conv2D(8, (3, 3), activation="relu", padding="same")(conv31)

        up1 = UpSampling2D()(conv32)
        conv41 = Conv2D(16, (3, 3), activation="relu", padding="same")(up1)
        conv42 = Conv2D(16, (3, 3), activation="relu", padding="same")(conv41)

        up2 = UpSampling2D()(conv42)
        conv51 = Conv2D(32, (3, 3), activation="relu", padding="same")(up2)
        conv52 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv51)

        decoded = Conv2D(3, (1, 1), activation="sigmoid", padding="same")(conv52)

        model = Model(inputs=input_layer, outputs=decoded)

        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=100000,  # 減衰が適用されるステップ数
            decay_rate=0.96,  # 減衰率
            staircase=False,  # True: 階段的な減衰, False: 連続的な減衰
        )

        adam = Adam(learning_rate=lr_schedule)

        model.compile(optimizer=adam, loss="mse")

        return model
