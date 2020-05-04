import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Embedding, LSTM, Dense, Dropout, BatchNormalization, Concatenate, Input
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l1, l2, l1_l2


def get_dnn_model(input_shape):
    # Define the model
    '''
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(128, activation="relu", kernel_regularizer=l1_l2(0.01, 0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu", kernel_regularizer=l1_l2(0.01, 0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu", kernel_regularizer=l1_l2(0.01, 0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4, activation="softmax"))
    '''
    x_input = Input(shape=input_shape)
    x1 = Dense(32, activation="relu", kernel_regularizer=l1_l2(0.01, 0.01))(x_input)
    x1 = BatchNormalization()(x1)
    x2 = Concatenate()([x_input, x1])
    x2 = Dropout(0.5)(x2)
    x3 = Dense(64, activation="relu", kernel_regularizer=l1_l2(0.01, 0.01))(x2)
    x3 = BatchNormalization()(x3)
    x4 = Concatenate()([x2, x3])
    x4 = Dropout(0.5)(x4)
    x5 = Dense(128, activation="relu", kernel_regularizer=l1_l2(0.01, 0.01))(x4)
    x5 = BatchNormalization()(x5)
    x6 = Concatenate()([x4, x5])
    x6 = Dropout(0.5)(x6)
    x7 = Dense(256, activation="relu", kernel_regularizer=l1_l2(0.01, 0.01))(x6)
    x7 = BatchNormalization()(x7)
    out = Dense(4, activation="softmax")(x7)

    model = keras.Model(inputs=[x_input], outputs=[out])

    opt = Adam(learning_rate=0.001)
    #opt = SGD(learning_rate=0.01, momentum=0.9)

    model.compile(optimizer=opt, loss=CategoricalCrossentropy(from_logits=True), metrics=["accuracy"])

    return model
