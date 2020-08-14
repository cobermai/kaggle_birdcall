import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self,                 
                 n_layers,
                 n_channels,
                 name = "generator",
                 kernel_regularizer = tf.keras.regularizers.l2(2e-4),
                 kernel_initializer = tf.keras.initializers.he_normal()):
        super(Generator, self).__init__()
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.model_name = name
        self.kernel_regularizer = None
        self.kernel_initializer = kernel_initializer 
        
    def build(self,input_shape):
        
        self.model_layers = []
        #Compression layers
        for i in range(self.n_layers):

                self.model_layers.append((tf.keras.layers.Conv2D(self.n_channels[i],
                        kernel_size=(3, 3),
                        strides=(2,2),
                        padding="same",
                        use_bias=False,
                        name=self.model_name+'_conv_' + str(i),
                        kernel_regularizer = self.kernel_regularizer,
                        kernel_initializer = self.kernel_initializer,
                        activation=None),0))
                self.model_layers.append((tf.keras.layers.BatchNormalization(axis=-1),1))
                self.model_layers.append((tf.keras.layers.Activation("relu"),0))
        #Decompression layers
        for i in range(self.n_layers+1):

            if i == self.n_layers:
                self.model_layers.append((tf.keras.layers.Conv2DTranspose(input_shape[-1],
                                                                 kernel_size=(1, 1),
                                                                 strides=(1, 1),
                                                                 padding="same",
                                                                 use_bias=False,
                                                                 name=self.model_name + '_conv_' + str(i),
                                                                 kernel_regularizer=None,
                                                                 kernel_initializer=self.kernel_initializer,
                                                                 activation=None), 0))
            else:
                self.model_layers.append((tf.keras.layers.Conv2DTranspose(self.n_channels[self.n_layers-1-i],
                                                                 kernel_size=(3, 3),
                                                                 strides=(2, 2),
                                                                 padding="same",
                                                                 use_bias=False,
                                                                 name=self.model_name + '_conv_tp_' + str(i),
                                                                 kernel_regularizer=self.kernel_regularizer,
                                                                 kernel_initializer=self.kernel_initializer,
                                                                 activation=None), 0))
                self.model_layers.append((tf.keras.layers.BatchNormalization(axis=-1), 1))
                self.model_layers.append((tf.keras.layers.Activation("relu"), 0))
    
    def call(self,input,training=False):
        x = input
        
        for layer in self.model_layers:
            if layer[1]==0:
                x = layer[0](x)
            elif layer[1]==1:
                x = layer[0](x,training)
        x = x[:,:input.shape[1],:input.shape[2],:]
        return x+input
