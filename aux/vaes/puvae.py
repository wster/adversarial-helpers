from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Layer, Dense, Conv2D, Flatten, Conv2DTranspose, Reshape, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error

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
        self.dense = Dense(7*7*32, activation='relu')
        self.deconv1 = Conv2DTranspose(64, (3,3), strides=2, padding='same', activation='relu')
        self.deconv2 = Conv2DTranspose(32, (3,3), strides=2, padding='same', activation='relu')
        self.deconv3 = Conv2DTranspose(1, (3,3), padding='same', activation='relu') # should be sigmoid but results in nan

    def call(self, inputs):
        x, y = inputs

        x1 = Concatenate()([x, y])
        x2 = self.dense(x1)
        x3 = Reshape((7,7,32))(x2)
        x4 = self.deconv1(x3)
        x5 = self.deconv2(x4)
        x6 = self.deconv3(x5)

        return x6

class PuVAE(Model):
    def __init__(self, latent_dim=32, use_kl_loss=True, **kwargs):
        super(PuVAE, self).__init__(**kwargs)
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()
        self.use_kl_loss = use_kl_loss

    def call(self, inputs):
        x, y = inputs
        z_mean, z_log_var, z = self.encoder([x, y])
        reconstructions = self.decoder([z, y])

        #kl_loss = K.mean(K.square(z_mean)) + K.mean(K.square(z_log_var)) - K.log(K.mean(K.square(z_log_var)) - 1)
        rc_loss = K.mean(mean_squared_error(x, reconstructions))
        #self.add_loss(rc_loss + kl_loss)
        self.add_loss(rc_loss)

        return reconstructions