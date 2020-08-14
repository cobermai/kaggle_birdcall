import tensorflow as tf
from models.layers.spectral_normalization import SpectralNormalization

class Discriminator(tf.keras.Model):
    def __init__(self,
                 n_layers,
                 n_channels,
                 strides,
                 name = "discriminator",
                 kernel_regularizer = None,
                 kernel_initializer = tf.keras.initializers.he_normal()):
        super(Discriminator, self).__init__()
        self.n_layers = n_layers
        self.strides = strides
        self.n_channels = n_channels
        self.model_name = name
        self.kernel_regularizer = None
        self.kernel_initializer = kernel_initializer 
        
    def build(self,input_shape):
        
        self.model_layers = []
        for i in range(self.n_layers):

            self.model_layers.append(SpectralNormalization(tf.keras.layers.Conv2D(self.n_channels[i],
                        kernel_size=self.strides[i]+1,
                        strides=self.strides[i],
                        padding="same",
                        use_bias=False,
                        name=self.model_name+'_conv_' + str(i),
                        kernel_regularizer = self.kernel_regularizer,
                        kernel_initializer = self.kernel_initializer,
                        activation='relu')))
            
        self.dense = SpectralNormalization(tf.keras.layers.Dense(1,
                                                 kernel_initializer = self.kernel_initializer,
                                                 use_bias = False,
                                                 name = self.model_name +"_dense",
                                                 activation = tf.nn.relu))
    
    
    def call(self,input,training=False):
        x = tf.image.per_image_standardization(input)
        
        for layer in self.model_layers:
            x = layer(x)

        x = self.dense(x)

        return x
