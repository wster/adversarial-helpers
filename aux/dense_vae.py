from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, mse

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class Encoder(Layer):
    def __init__(self, latent_dim=32, intermediate_dim=64, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.w1 = Dense(intermediate_dim, activation='relu', name='encoder_w1')
        self.w_mean = Dense(latent_dim, name='encoder_w_mean')
        self.w_var = Dense(latent_dim, name='encoder_w_var')
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.w1(inputs)
        z_mean = self.w_mean(x)
        z_log_var = self.w_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(Layer):
    def __init__(self, original_dim, intermediate_dim=64, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.w1 = Dense(intermediate_dim, activation='relu', name='decoder_w1')
        self.w2 = Dense(original_dim, activation='sigmoid', name='decoder_w2')

    def call(self, inputs):
        x = self.w1(inputs)
        return self.w2(x)

class DenseVAE(Model):
    def __init__(self, original_dim, intermediate_dim=164, latent_dim=32, **kwargs):
        super(DenseVAE, self).__init__(**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

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
        _, _, z = self.encoder(inputs)
        reconstructions = self.decoder(z)
        return reconstructions