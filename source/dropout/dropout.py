import keras.backend as K
import numpy as np
from keras.layers import SpatialDropout1D
from keras.layers import Input, Dropout, LSTM


def normal_dropout():
    input_layer = Input(shape=(4, 6), dtype='float')
    dropout = Dropout(0.5)(input_layer, training=1)
    f = K.function(inputs=[input_layer], outputs=[dropout])
    data = np.random.random(size=(4, 6))
    result = f([data])
    print('before:')
    print(data)
    print('after:')
    print(result[0])


def spatial1d_dropout():
    input_layer = Input(shape=(4, 6), dtype='float')
    dropout = SpatialDropout1D(0.5)(input_layer, training=1)
    f = K.function(inputs=[input_layer], outputs=[dropout])
    data = np.random.random(size=(1, 4, 6))
    result = f([data])
    print('before:')
    print(data)
    print('after:')
    print(result[0])


def variational_dropout():
    input_layer = Input(shape=(4, 6), dtype='float')
    dropout = LSTM(7, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)\
        (input_layer, training=1)
    f = K.function(inputs=[input_layer], outputs=[dropout])
    data = np.random.random(size=(1, 4, 6))
    result = f([data])
    print('before:')
    print(data)
    print('after:')
    print(result[0])


if __name__ == '__main__':
    #normal_dropout()
    #spatial1d_dropout()
    variational_dropout()


