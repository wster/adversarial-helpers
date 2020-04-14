from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

class DenseAE(Model):
    def __init__(self, original_dim, intermediate_dim, latent_dim, **kwargs):
        super(DenseAE, self).__init__(**kwargs)
        self.intermediate1 = Dense(intermediate_dim, activation='relu')
        self.latent = Dense(latent_dim, activation='relu')
        self.intermediate2 = Dense(intermediate_dim, activation='relu')
        self.reconstruction = Dense(original_dim, activation='sigmoid')
    
    def call(self, inputs):
        intermediate1 = self.intermediate1(inputs)
        latent = self.latent(intermediate1)
        intermediate2 = self.intermediate2(latent)
        reconstruction = self.reconstruction(intermediate2)
        return reconstruction