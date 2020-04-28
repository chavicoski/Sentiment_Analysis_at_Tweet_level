import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD


def get_dnn_model(input_shape):
    # Define the model
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, activation="softmax"))

    opt = Adam(learning_rate=0.001)
    #opt = SGD(learning_rate=0.01, momentum=0.9)

    model.compile(optimizer=opt, loss=CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    return model
