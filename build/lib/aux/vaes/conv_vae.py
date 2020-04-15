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

        self.w_mean = Dense(latent_dim, name='encoder_w_mean')
        self.w_var = Dense(latent_dim, name='encoder_w_var')

        self.sampling = Sampling()

    def call(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = Flatten()(x3)
        x5 = self.dense1(x4)
        x6 = self.dense2(x5)
        
        z_mean = self.w_mean(x6)
        z_log_var = self.w_var(x6)
        z = self.sampling((z_mean, z_log_var))

        return z_mean, z_log_var, z

class Decoder(Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense = Dense(7*7*32, activation='relu')
        self.deconv1 = Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')
        self.deconv2 = Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')
        self.deconv3 = Conv2DTranspose(1, (3,3), padding='same', activation='relu')

    def call(self, inputs):
        x1 = self.dense(inputs)
        x2 = Reshape((7,7,32))(x1)
        x3 = self.deconv1(x2)
        x4 = self.deconv2(x3)
        x5 = self.deconv3(x4)

        return x5

class ConvVAE(Model):
    def __init__(self, original_dim, latent_dim=32, targets=None, **kwargs):
        super(ConvVAE, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.targets = targets
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()

    def call(self, inputs):
        """
        z_mean, z_var, z = self.encoder(inputs)
        reconstructions = self.decoder(z)
        reconstruction_loss = mse(inputs, reconstructions)
        reconstruction_loss *= self.original_dim
        
        kl_loss = - 0.5 * K.mean(z_var - K.square(z_mean) - K.exp(z_var) + 1)
        kl_loss = 1 + z_var - K.square(z_mean) - K.exp(z_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        self.add_loss(K.mean(kl_loss + reconstruction_loss))
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructions = self.decoder(z)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)

        targets = inputs if self.targets is None else self.targets
        reconstruction_loss = mse(reconstructions, targets)

        self.add_loss(kl_loss + reconstruction_loss)

        return reconstructions