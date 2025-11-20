import tensorflow as tf
from tensorflow import keras
import numpy as np

B_fixed = tf.constant([
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0],
    [9.0, 10.0]
], dtype=tf.float32)

# Test with sample data
# A_sample = tf.random.normal((4, 2))  # Batch size 4

A_sample = tf.constant([[9, 10],
                 [3, 4],
                 [8, 6]], dtype=tf.float32)  # Shape (3, 2)

# Reshape for broadcasting
A_expanded = tf.expand_dims(A_sample, axis=2)  # Shape (batch, 2, 1)
B_expanded = tf.expand_dims(tf.transpose(B_fixed), axis=0)  # Shape (1, 2, 5)

# Subtraction with broadcasting
C = A_expanded - B_expanded  # Shape (batch, 2, 5)

def custom_activation(x):
    a=tf.sigmoid(x)
    b=tf.sigmoid(-x)
    return a*b

max_pool_1d = keras.layers.MaxPooling1D(pool_size=2,
                                                strides=1, padding="valid")
C1 = custom_activation(C)

splittedChannels = tf.split(C1,[1,1],axis=1)

min_channels=tf.keras.layers.minimum([splittedChannels[0],splittedChannels[1]])

max_pool_2d = keras.layers.MaxPooling2D(pool_size=(1, 5),
                                                strides=1, padding="valid")
x = tf.expand_dims(min_channels, axis=-1)
poolValue2 = max_pool_2d(x)

poolValue3 = tf.keras.layers.Reshape((-1,))(poolValue2)

print("Output shape:", poolValue3)  # Should be (4, 2, 5)
print("!!!")

