from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Layer, Dense, Conv2D, Flatten, Conv2DTranspose, Reshape, Concatenate, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error

class Sampling(Layer):
    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        sigma_epsilon = 0.1

        if 'training' in kwargs:
            training = kwargs.get("training")
            sigma_epsilon = 1.0 if training else 0.1

        epsilon = K.random_normal(shape=(batch, dim), stddev=sigma_epsilon)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class Encoder(Layer):
    def __init__(self, latent_dim=32, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.conv1 = Conv2D(32, (7,7), dilation_rate=2, activation='relu', padding='same')
        self.conv2 = Conv2D(32, (7,7), dilation_rate=2, activation='relu', padding='same')
        self.conv3 = Conv2D(32, (7,7), dilation_rate=2, activation='relu', padding='same')

        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(1024, activation='relu')

        self.dense3 = Dense(latent_dim, activation='relu', name='encoder_w_mean')
        self.dense4 = Dense(latent_dim, activation='softplus', name='encoder_w_var')

        self.sampling = Sampling()

    def call(self, inputs):
        x, y = inputs

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = Flatten()(x3)
        x5 = Concatenate()([x4, y])
        x6 = self.dense1(x5)
        x7 = self.dense2(x6)

        z_mean = self.dense3(x7)
        z_log_var = self.dense4(x7)
        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z

class Decoder(Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense = Dense(512, activation='relu')
        self.deconv1 = Conv2DTranspose(32, (7,7), strides=2, padding='valid', activation='relu')
        self.deconv2 = Conv2DTranspose(32, (7,7), strides=2, padding='same', activation='relu')
        self.deconv3 = Conv2DTranspose(32, (7,7), strides=2, padding='same', activation='relu')
        self.deconv4 = Conv2DTranspose(1, (3,3), padding='same', activation='sigmoid')

    def call(self, inputs):
        x, y = inputs

        x1 = Concatenate()([x, y])
        x2 = self.dense(x1)
        x3 = Reshape((1,1,512))(x2)
        x4 = self.deconv1(x3)
        x5 = self.deconv2(x4)
        x6 = self.deconv3(x5)
        x7 = self.deconv4(x6)

        return x7

class CIFAREncoder(Layer):
    def __init__(self, latent_dim=32, **kwargs):
        super(CIFAREncoder, self).__init__(**kwargs)

        self.conv1 = Conv2D(3, (2,2), padding='same', activation='relu')
        self.conv2 = Conv2D(32, (2,2), dilation_rate=2, padding='same', activation='relu')
        self.conv3 = Conv2D(32, (2,2), dilation_rate=2, padding='same', activation='relu')
        self.conv4 = Conv2D(32, (2,2), dilation_rate=2, padding='same', activation='relu')

        self.dense1 = Dense(1024, activation='relu')

        self.dense2 = Dense(latent_dim, activation='relu', name='encoder_w_mean')
        self.dense3 = Dense(latent_dim, activation='softplus', name='encoder_w_var')

        self.sampling = Sampling()

    def call(self, inputs):
        x, y = inputs

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = Flatten()(x4)
        x6 = Concatenate()([x5, y])
        x7 = self.dense1(x6)

        z_mean = self.dense2(x7)
        z_log_var = self.dense3(x7)
        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z

class CIFARDecoder(Layer):
    def __init__(self, **kwargs):
        super(CIFARDecoder, self).__init__(**kwargs)
        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(8192, activation='relu')
        self.deconv1 = Conv2DTranspose(32, (2,2), padding='same', activation='relu')
        self.deconv2 = Conv2DTranspose(32, (2,2), padding='same', activation='relu')
        self.deconv3 = Conv2DTranspose(32, (3,3), strides=2, padding='valid', activation='relu')
        self.conv1 = Conv2D(3, (2,2), activation='sigmoid')

    def call(self, inputs):
        x, y = inputs

        x1 = Concatenate()([x, y])
        x2 = self.dense1(x1)
        x3 = self.dense2(x2)
        x4 = Reshape((16,16,32))(x3)
        x5 = self.deconv1(x4)
        x6 = self.deconv2(x5)
        x7 = self.deconv3(x6)
        x8 = self.conv1(x7)

        return x8

class PuVAE(Model):
    def __init__(self, latent_dim=32, dataset='mnist', **kwargs):
        super(PuVAE, self).__init__(**kwargs)
        self.encoder = Encoder(latent_dim) if dataset == 'mnist' else CIFAREncoder(latent_dim)
        self.decoder = Decoder() if dataset == 'mnist' else CIFARDecoder()

    def call(self, inputs):
        x, y = inputs
        z_mean, z_log_var, z = self.encoder([x, y])
        reconstructions = self.decoder([z, y])
        return (z_mean, z_log_var, reconstructions)