import tensorflow as tf
from keras import layers
from keras.models import Model

from mltu.tensorflow.model_utils import residual_block, activation_layer


def train_model(input_dim, output_dim, activation="leaky_relu", dropout=0.2, rnn_units=128, m_cnt=2,f_mel=False):
    
    inputs = layers.Input(shape=input_dim, name="input")

    # expand dims to add channel dimension
    input = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)

    # Convolution layer 1
    if f_mel == False:
        x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
        #x = layers.Conv2D(filters=40, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
    else:
        x = layers.Conv2D(filters=32, kernel_size=[11, 41], strides=[2, 2], padding="same", use_bias=False)(input)
        #x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[2, 2], padding="same", use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")

    # Convolution layer 2
    if f_mel == False:
        x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
        #x = layers.Conv2D(filters=40, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
    else:
        x = layers.Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2], padding="same", use_bias=False)(x)
        #x = layers.Conv2D(filters=32, kernel_size=[11, 11], strides=[1, 2], padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = activation_layer(x, activation="leaky_relu")
    
    # Reshape the resulted volume to feed the RNNs layers
    x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

    # RNN layers
    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))(x)

    # Dense layer
    #x = layers.Dense(256)(x)
    #x = layers.Dense(1024)(x)       # changed by nishi 20223.8.14
    x = layers.Dense(rnn_units * m_cnt)(x)       # changed by nishi 20223.8.16
    x = activation_layer(x, activation="leaky_relu")
    x = layers.Dropout(dropout)(x)

    # Classification layer
    output = layers.Dense(output_dim + 1, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model
