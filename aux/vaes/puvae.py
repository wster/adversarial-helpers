from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Layer, Dense, Conv2D, Flatten, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class Encoder(Layer):
    def __init__(self, latent_dim=32, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.conv1 = Conv2D(32, (7,7), dilation_rate=2, activation='relu', padding='same')
        self.conv2 = Conv2D(32, (7,7), dilation_rate=2, activation='relu', padding='same')
        self.conv3 = Conv2D(32, (7,7), dilation_rate=2, activation='relu', padding='same')

        self.dense1 = Dense(1024, activation='relu')
        self.dense2 = Dense(1024, activation='relu')

        self.dense3 = Dense(latent_dim, name='encoder_w_mean')
        self.dense4 = Dense(latent_dim, name='encoder_w_var') # should maybe run this through softplus

        self.sampling = Sampling()

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = Flatten()(x3)
        x5 = self.dense1(x4)
        x6 = self.dense2(x5)
        
        z_mean = self.dense3(x6)
        z_log_var = self.dense4(x6)
        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z

class Decoder(Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense = Dense(7*7*32, activation='relu')
        self.deconv1 = Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')
        self.deconv2 = Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')
        self.deconv3 = Conv2DTranspose(1, (3,3), padding='same', activation='sigmoid')

    def call(self, inputs):
        x1 = self.dense(inputs)
        x2 = Reshape((7,7,32))(x1)
        x3 = self.deconv1(x2)
        x4 = self.deconv2(x3)
        x5 = self.deconv3(x4)

        return x5

class PuVAE(Model):
    def __init__(self, latent_dim=32, **kwargs):
        super(PuVAE, self).__init__(**kwargs)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    def call(self, inputs):
        #z_mean, z_log_var, z = self.encoder(inputs) <---- might come in handy later
        _, _, z = self.encoder(inputs)
        reconstructions = self.decoder(z)
        return reconstructions