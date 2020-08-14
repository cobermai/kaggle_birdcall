
from models.layers.dense_moe import DenseMoE
import tensorflow as tf

def basic_dmoe_cnn_mnist():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(DenseMoE(64, n_experts=10, expert_activation='relu', gating_activation='softmax'))
    model.add(tf.keras.layers.Dense(10))

    return model

        
        

