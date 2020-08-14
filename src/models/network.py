from models.layers.dense_moe import DenseMoE
import tensorflow as tf


class Classifier(tf.keras.Model):
    def __init__(self,
                 network_block,
                 n_blocks,
                 n_layers,
                 strides,
                 channel_base,
                 n_classes,
                 init_ch,
                 init_ksize,
                 init_stride,
                 use_max_pool = True,
                 kernel_regularizer = tf.keras.regularizers.l2(2e-4),
                 kernel_initializer = tf.keras.initializers.he_normal(),
                 dropout=0.2):
        super(Network, self).__init__()
        self.network_block = network_block
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.strides = strides
        self.channel_base = channel_base
        self.n_classes = n_classes        
        self.dropout = dropout 
        self.init_ch = init_ch
        self.init_ksize = init_ksize
        self.init_stride = init_stride
        self.use_max_pool = use_max_pool
        self.kernel_regularizer = kernel_regularizer
        self.kernel_initializer = kernel_initializer 
        
    def build(self,input_shape):
        
        self.init_conv = tf.keras.layers.Conv2D(self.init_ch,
                                                self.init_ksize,
                                                self.init_stride,
                                                padding = "same",
                                                use_bias = False,
                                                name = 'initial_conv',
                                                kernel_regularizer = self.kernel_regularizer,
                                                kernel_initializer = self.kernel_initializer)
        
        self.init_bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.init_relu = tf.keras.layers.Activation("relu")
        
        if self.use_max_pool:
            self.init_max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), 
                                                       strides=(2, 2),
                                                       padding="same")
        
        self.network_blocks = []
        for i_block in range(self.n_blocks):
            self.network_blocks.append(self.network_block(self.n_layers[i_block],
                                                          self.channel_base[i_block],
                                                          stride = self.strides[i_block],
                                                          kernel_regularizer = self.kernel_regularizer,
                                                          kernel_initializer = self.kernel_initializer))
            
        self.last_bn = tf.keras.layers.BatchNormalization(axis=-1)
        self.last_relu = tf.keras.layers.Activation("relu")
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.keras.layers.Dense(self.n_classes,
                                                name = 'dense_layer',
                                                kernel_regularizer = self.kernel_regularizer,
                                                kernel_initializer = self.kernel_initializer)
            
    def call(self,input,training=False):
        """Returns logits"""
        x = self.init_conv(input)
        x = self.init_bn(x,training)
        x = self.init_relu(x)
        if self.use_max_pool:
            x = self.init_max_pool(x)
        
        for block in self.network_blocks:
            x = block(x,training)
            
        x = self.last_bn(x,training)
        x = self.last_relu(x)
        x = self.avg_pool(x)
        x = self.dense(x)
        
        return x
    
