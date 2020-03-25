from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense


class neural_network:
    def __init__(self):
        pass

    def build(self, Witdh, Height, Depth, total_classes, Save_Weigths_Path=None):
        model = Sequential()

        model.add(Convolution2D(20, 5, 5,  border_mode="same",
                                input_shape=(Height, Witdh, Depth)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="th"))

        model.add(Convolution2D(50, 5, 5,  border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),  dim_ordering="th"))

        model.add(Convolution2D(100, 5, 5,  border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2),  dim_ordering="th"))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(total_classes))
        model.add(Activation("softmax"))
        
        return model
    




