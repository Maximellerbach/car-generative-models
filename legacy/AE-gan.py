import os
import time
from glob import glob

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    Reshape,
    ZeroPadding2D,
    multiply,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import (
    AveragePooling2D,
    Conv2D,
    DepthwiseConv2D,
    Conv2DTranspose,
    MaxPooling2D,
    UpSampling2D,
    ZeroPadding2D,
)
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from tqdm import tqdm


class AEGAN:
    def __init__(self):

        self.img_rows = 100
        self.img_cols = 150
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 250

        optimizer = Adam(0.0001, 0.5)

        # uncomment to build ae, encoder, decoder
        self.encoder, self.decoder, self.ae = self.create_ae()

        # uncomment to load ae, encoder, decoder
        self.encoder = load_model("encoder.h5")
        self.decoder = load_model("decoder.h5")

        """
        inp = Input((self.img_shape))
        y = self.encoder(inp)
        x = self.decoder(y)
        self.ae = Model(inp, x)
        
        X_train = self.load_dataset('C:\\Users\\maxim\\Desktop\\car_img\\*')
        self.ae.compile(optimizer=Adam(), loss='mse', metrics=['accuracy'])
        self.ae.fit(X_train, X_train, batch_size=32, epochs=5)
        
        self.encoder.save('encoder.h5')
        self.decoder.save('decoder.h5')
        """

        # uncomment to build discriminator, generator
        self.discriminator = self.create_discriminator()
        self.discriminator.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        self.generator = self.create_generator()

        """
        # uncomment to load discriminator, generator
        self.discriminator = load_model('vroum\\vroumdis.h5')
        self.discriminator.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        self.generator = load_model('vroum\\vroumgen.h5')
        """
        # the combined model take an image as input and output validity from 0 to 1
        # note that in the combined model, the discriminator is not trainable
        self.discriminator.trainable = False

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss="binary_crossentropy", optimizer=optimizer)

    def create_ae(self):

        encoder = Sequential()

        encoder.add(Conv2D(16, kernel_size=3, input_shape=self.img_shape, padding="same"))
        encoder.add(Activation("relu"))
        encoder.add(BatchNormalization())
        encoder.add(MaxPooling2D())
        encoder.add(Dropout(0.1))

        encoder.add(DepthwiseConv2D(3, padding="same"))
        encoder.add(Activation("relu"))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(32, kernel_size=3, padding="same"))
        encoder.add(MaxPooling2D())
        encoder.add(Dropout(0.1))

        encoder.add(DepthwiseConv2D(3, padding="same"))
        encoder.add(Activation("relu"))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(64, kernel_size=3, padding="same"))
        encoder.add(MaxPooling2D())
        encoder.add(Dropout(0.1))

        encoder.add(DepthwiseConv2D(3, padding="same"))
        encoder.add(Activation("relu"))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(128, kernel_size=3, padding="same"))
        encoder.add(MaxPooling2D())
        encoder.add(Dropout(0.1))

        encoder.add(DepthwiseConv2D(3, padding="same"))
        encoder.add(Activation("relu"))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(256, kernel_size=3, padding="same"))
        encoder.add(Dropout(0.1))

        encoder.add(DepthwiseConv2D(3, padding="same"))
        encoder.add(Activation("relu"))
        encoder.add(BatchNormalization())
        encoder.add(Conv2D(256, kernel_size=3, padding="same"))

        encoder.add(Flatten())

        encoder.add(Dense(self.latent_dim, activation="relu", use_bias=False))
        encoder.add(BatchNormalization())
        encoder.add(Dense(self.latent_dim, activation="relu", use_bias=False))
        encoder.add(BatchNormalization())
        encoder.add(Dense(self.latent_dim, activation="relu", use_bias=False))

        encoder.summary
        ############

        decoder = Sequential()

        decoder.add(Dense(self.latent_dim, activation="relu", use_bias=False))
        encoder.add(BatchNormalization())

        decoder.add(Dense(self.latent_dim, activation="relu", use_bias=False))
        encoder.add(BatchNormalization())

        decoder.add(Dense(6 * 9 * 8, activation="relu", use_bias=False, input_dim=(self.latent_dim)))
        decoder.add(BatchNormalization(momentum=0.8))

        decoder.add(Reshape((6, 9, 8)))

        decoder.add(Dense(256, use_bias=False))
        decoder.add(Activation("relu"))
        decoder.add(BatchNormalization(momentum=0.8))

        decoder.add(Conv2D(256, kernel_size=(3, 3), padding="same"))
        decoder.add(Activation("relu"))
        decoder.add(BatchNormalization(momentum=0.8))

        decoder.add(UpSampling2D())

        decoder.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
        decoder.add(Activation("relu"))
        decoder.add(BatchNormalization(momentum=0.8))

        decoder.add(UpSampling2D())
        decoder.add(ZeroPadding2D(padding=((0, 1), (0, 1))))

        decoder.add(Conv2D(64, kernel_size=(3, 3), padding="same"))
        decoder.add(Activation("relu"))
        decoder.add(BatchNormalization(momentum=0.8))

        decoder.add(UpSampling2D())
        decoder.add(ZeroPadding2D(padding=(0, (1, 0))))

        decoder.add(Conv2D(32, kernel_size=(3, 3), padding="same"))
        decoder.add(Activation("relu"))
        decoder.add(BatchNormalization(momentum=0.8))

        decoder.add(UpSampling2D())

        decoder.add(Conv2D(16, kernel_size=(3, 3), padding="same"))
        decoder.add(Activation("relu"))
        decoder.add(BatchNormalization(momentum=0.8))

        decoder.add(Dense(self.channels))
        decoder.add(Activation("tanh"))

        inp = Input(shape=(self.img_shape))
        y = encoder(inp)
        x = decoder(y)

        ae = Model(inp, x)
        ae.summary()

        return encoder, decoder, ae

    def create_generator(self):

        model = Sequential()

        model.add(Dense(6 * 9 * 8, activation="relu", use_bias=False, input_dim=(self.latent_dim)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.latent_dim * 4, activation="relu", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(self.latent_dim, activation="relu", use_bias=False))
        model.add(BatchNormalization(momentum=0.8))

        noise = Input((self.latent_dim,))
        y = model(noise)

        x = self.decoder(y)

        gen = Model(noise, x)
        gen.summary()

        return gen

    def create_discriminator(self):

        model = Sequential()

        model.add(Dense(1, activation="sigmoid"))

        img = Input(shape=self.img_shape)
        y = self.encoder(img)
        validity = model(y)

        model = Model(img, validity)
        model.summary()

        return model

    def train(self, batch_size=128, save_interval=50, save_img_interval=50):

        # get dataset
        X_train = self.load_dataset("C:\\Users\\maxim\\Desktop\\car_img\\*")

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
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator with generated images and real images
            d_loss_r = self.discriminator.train_on_batch(imgs, ones)
            d_loss_f = self.discriminator.train_on_batch(gen_imgs, zeros)
            d_loss = np.add(d_loss_r, d_loss_f) * 0.5

            # Trains the generator to fool the discriminator
            g_loss = self.combined.train_on_batch(noise, ones)

            # print loss and accuracy of both trains
            print("%d D loss: %f, acc.: %.2f%% G loss: %f" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            if epoch % save_img_interval == 0:
                self.save_imgs(epoch)

            if epoch % save_interval == 0:

                # self.discriminator.save('gan\\vroum\\vroumdis_'+str(epoch)+'.h5')
                # self.generator.save('gan\\vroum\\vroumgen_'+str(epoch)+'.h5')

                self.discriminator.save("vroum\\vroumdis2.h5")
                self.generator.save("vroum\\vroumgen2.h5")

    def save_imgs(self, e):

        gen_img = self.generator.predict(self.noise_pred)
        # confidence = self.discriminator.predict(gen_img)

        # Rescale image to 0 - 255
        gen_img = (0.5 * gen_img + 0.5) * 255

        cv2.imwrite("car\\%f_%f.png" % (time.time(), e), gen_img[0])

    def load_dataset(self, path):

        try:
            # try to load existing X_train
            X_train = np.load("X_train.npy")
            print("loaded dataset")

        except:
            # else, build X_train and save it
            X_train = []
            dos = glob(path)

            for i in tqdm(dos):
                img = cv2.imread(i)
                img = cv2.resize(img, (self.img_cols, self.img_rows))

                X_train.append(img)

            cv2.destroyAllWindows()
            X_train = np.array(X_train)

            # Rescale dataset to -1 - 1
            X_train = X_train / 127.5 - 1

            np.save("X_train.npy", X_train)
            print("created dataset")

        return X_train


if __name__ == "__main__":

    ae = AEGAN()
    ae.train(batch_size=32, save_interval=5000, save_img_interval=25)
