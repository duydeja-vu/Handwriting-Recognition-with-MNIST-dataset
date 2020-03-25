from keras.datasets import mnist
from keras.utils import np_utils
from keras import optimizers
from model import neural_network



def create_model():
    clf = neural_network()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = clf.build(28, 28, 1, 10)
    model.compile(loss="categorical_crossentropy",
                  optimizer=sgd, metrics=["accuracy"])
    return model


def train_and_test(model, batch_size, epoch):
    image_width = image_height = 28
    image_channel = 1
    print('Loading MNIST Dataset...')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(
        (X_train.shape[0], image_width, image_height, image_channel)).astype('float32')
    X_test = X_test.reshape(
        (X_test.shape[0], image_width, image_height, image_channel)).astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    model.fit(X_train, y_train, validation_data=(
        X_test, y_test), epochs=epoch, batch_size=batch_size)
    scores = model.evaluate(X_test, y_test, verbose=0)
    return scores





model = create_model()
batch_size = 128
epoch = 10
acc = train_and_test(model, batch_size, epoch)
print(acc)
