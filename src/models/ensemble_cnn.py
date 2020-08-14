
from models.layers.dense_moe import DenseMoE
import tensorflow as tf


def model_base(x, wd, name=''):

    x = tf.keras.layers.Convolution2D(16, kernel_size=3, strides=1, data_format='channels_last',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=tf.keras.regularizers.l2(wd))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    # next layer has 32 convolution filters,
    x = tf.keras.layers.Convolution2D(32, kernel_size=3, strides=2,
                       kernel_regularizer=tf.keras.regularizers.l2(wd))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Convolution2D(32, kernel_size=3, strides=1,
                       kernel_regularizer=tf.keras.regularizers.l2(wd))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    # next layer has 64 convolution filters
    x = tf.keras.layers.Convolution2D(64, kernel_size=3, strides=2,
                       kernel_regularizer=tf.keras.regularizers.l2(wd))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Convolution2D(64, kernel_size=3, strides=1,
                       kernel_regularizer=tf.keras.regularizers.l2(wd))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')(x)

    return x

def ensemble_cnn_mnist(x,n_models,wd,n_classes):

    x_tensor = tf.keras.layers.Input(x.shape[1:], name='Input')
    outputs = []
    for i_model in range(n_models):
        output_i = model_base(x_tensor,wd)
        outputs.append(output_i)
    x = tf.keras.layers.Concatenate(axis=-1)(outputs)
    # Softmax
    x = DenseMoE(64, n_experts=10, expert_activation='relu', gating_activation='softmax')(x)
    output = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=x_tensor, outputs=output)
    return model