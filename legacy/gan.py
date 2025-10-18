import os
import time
from glob import glob

import cv2
import numpy as np
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
    SeparableConv2D,
    UpSampling2D,
    ZeroPadding2D,
)
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm


class DCGAN:
    def __init__(self, pretrained=True):

        self.img_rows = 100
        self.img_cols = 150
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        if pretrained:
            self.discriminator = load_model("vroum\\vroumdis2.h5")
            self.generator = load_model("vroum\\vroumgen2.h5")

            self.discriminator.trainable = True
            self.generator.trainable = True

        else:
            self.discriminator = self.create_discriminator()
            self.generator = self.create_generator()

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        valid = self.discriminator(img)

        self.combined = Model(z, valid)

        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)
        self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        self.generator.summary()
        self.discriminator.summary()
        self.combined.summary()

    def create_generator(self):

        inp = Input(shape=(self.latent_dim,))
        x = Dense(6 * 9 * 256, use_bias=False)(inp)
        x = LeakyReLU(alpha=0.2)(x)

        x = Reshape((6, 9, 256))(x)

        x = Conv2D(256, kernel_size=3, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = UpSampling2D()(x)

        x = Conv2D(128, kernel_size=3, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = UpSampling2D()(x)
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)

        x = Conv2D(64, kernel_size=3, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = UpSampling2D()(x)
        x = ZeroPadding2D(padding=(0, (1, 0)))(x)

        x = Conv2D(32, kernel_size=3, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = UpSampling2D()(x)

        x = Conv2D(16, kernel_size=3, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(self.channels, kernel_size=3, padding="same", activation="tanh", use_bias=False)(x)

        model = Model(inp, x, name="gen")
        model.summary()
        return model

    def create_discriminator(self):

        inp = Input(shape=self.img_shape)

        x = Conv2D(32, kernel_size=5, strides=2, padding="same", use_bias=False)(inp)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = AveragePooling2D()(x)
        x = Dropout(0.1)(x)

        x = Conv2D(64, kernel_size=5, strides=2, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = AveragePooling2D()(x)
        x = Dropout(0.1)(x)

        x = Conv2D(128, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.1)(x)

        x = SeparableConv2D(512, kernel_size=3, strides=1, padding="same", use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.1)(x)

        x = Conv2D(32, kernel_size=3, strides=1, use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Dropout(0.4)(x)
        x = Flatten()(x)

        validity = Dense(1, activation="sigmoid", use_bias=False)(x)

        model = Model(inp, validity, name="dis")
        model.summary()
        return model

    def train(self, batch_size=128, save_interval=50, save_img_interval=50):

        # get dataset
        X_train = self.load_dataset("car_img\\*")

        # ones = label for real images
        # zeros = label for fake images
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))

        # create some noise to track AI's progression
        self.noise_pred = np.random.normal(0, 1, (1, self.latent_dim))

        epoch = 0
        while 1:
            epoch += 1

            # Select a random batch of images in dataset
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator(noise, training=False)

            # Train the discriminator with generated images and real images
            self.discriminator.trainable = True
            d_loss_r = self.discriminator.train_on_batch(imgs, ones)
            d_loss_f = self.discriminator.train_on_batch(gen_imgs, zeros)
            d_loss = np.add(d_loss_r, d_loss_f) * 0.5

            # Trains the generator to fool the discriminator
            self.combined.layers[2].trainable = False
            g_loss = self.combined.train_on_batch(noise, ones)

            # print loss and accuracy of both trains
            print(f"{epoch} D loss: {d_loss[0]}, acc: {100*d_loss[1]}% G loss: {g_loss}")

            if epoch % save_img_interval == 0:
                self.save_imgs(epoch)

            if epoch % save_interval == 0:

                # self.discriminator.save('gan\\vroum\\vroumdis_'+str(epoch)+'.h5')
                # self.generator.save('gan\\vroum\\vroumgen_'+str(epoch)+'.h5')

                self.generator.save("vroum\\vroumgen2.h5")
                self.discriminator.save("vroum\\vroumdis2.h5")

    def save_imgs(self, e):

        gen_img = self.generator(self.noise_pred).numpy()
        # confidence = self.discriminator.predict(gen_img)

        # Rescale image to 0 - 255
        gen_img = (0.5 * gen_img + 0.5) * 255

        cv2.imwrite("car\\%f_%d.png" % (time.time(), e), gen_img[0])

    def load_dataset(self, path):

        try:
            # try to load existing X_train
            X_train = np.load("X_train.npy")
            print("loaded dataset")

        except FileNotFoundError:
            # else, build X_train and save it
            X_train = []
            dos = glob(path)

            for i in tqdm(dos):
                img = cv2.imread(i)
                img = cv2.resize(img, (self.img_cols, self.img_rows))

                X_train.append(img)

            X_train = np.array(X_train)

            # Rescale dataset to -1 - 1
            X_train = X_train / 127.5 - 1

            np.save("X_train.npy", X_train)
            print("created dataset")

        return X_train


if __name__ == "__main__":
    cgan = DCGAN(pretrained=False)
    cgan.train(batch_size=32, save_interval=100, save_img_interval=25)
