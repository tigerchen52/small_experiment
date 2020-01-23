import keras.backend as K
from keras.models import Model
from keras.preprocessing import sequence
from keras.layers import Input, concatenate
from keras.layers import Dense, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras import optimizers

# set parameters:
batch_size = 32
embedding_dims = 200
filters = 250
kernel_size = 3
epochs = 2
kernel_size_list = [2, 3, 4, 5]

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=top_words)
# pad dataset to a maximum review length in words
max_words = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_words)
x_test = sequence.pad_sequences(x_test, maxlen=max_words)

# Build model
sentence = Input(batch_shape=(None, max_words), dtype='int32', name='sentence')
embedding_layer = Embedding(top_words, embedding_dims, input_length=max_words)
sent_embed = embedding_layer(sentence)

# use multi window-size cnn
cnn_result = []
for kernel_size in kernel_size_list:
    conv_layer = Conv1D(filters, kernel_size, padding='valid', activation='relu')
    sent_conv = conv_layer(sent_embed)
    sent_pooling = GlobalMaxPooling1D()(sent_conv)
    cnn_result.append(sent_pooling)
cnn_result = concatenate(cnn_result)

sent_repre = Dense(250)(cnn_result)
sent_repre = Activation('relu')(sent_repre)
sent_repre = Dense(1)(sent_repre)
pred = Activation('sigmoid')(sent_repre)
model = Model(inputs=sentence, outputs=pred)
#adam = optimizers.adam(lr=0.0005)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
