import keras

# import numpy as np
# import matplotlib.pyplot as plt
# import keras.backend as k

from keras.models import Sequential
from keras.layers import Dense, convolutional, Activation, Flatten, pooling
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from eye_detector_util import load_data


def get_train_test_data():
    data, label = load_data()
    # train_x, test_x, train_y, test_y = train_test_split(data, label, shuffle=True)
    train_x, test_x, train_y, test_y = train_test_split(
        data, label, shuffle=True, stratify=label
    )
    return train_x, test_x, train_y, test_y


def cnn(mode="Train"):
    try:
        print("Loading model")
        model = keras.models.load_model("eye_detector.save")

    except:
        print("Loading model failed: Initiating model")
        model = Sequential(
            [
                convolutional.Conv2D(16, 15, input_shape=(512, 512, 1)),
                pooling.MaxPooling2D(pool_size=4),
                convolutional.Conv2D(16, 7),
                Activation("tanh"),
                BatchNormalization(),
                convolutional.Conv2D(16, 7),
                Activation("tanh"),
                BatchNormalization(),
                convolutional.Conv2D(16, 7),
                Activation("tanh"),
                BatchNormalization(),
                pooling.MaxPooling2D(pool_size=4),
                convolutional.Conv2D(16, 3),
                Activation("tanh"),
                BatchNormalization(),
                pooling.MaxPooling2D(pool_size=4),
                Flatten(),
                Dense(16, activation="tanh"),
                BatchNormalization(),
                Dense(1, activation="tanh"),
            ]
        )

    model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["acc"])

    train_x, test_x, train_y, test_y = get_train_test_data()
    if mode == "Test":
        print(model.evaluate(test_x, test_y))
        print(model.metrics_names)

    if mode == "Train":
        best_loss = float("inf")
        history = model.fit(train_x, train_y, epochs=5)

        print(history)
        model.save("eye_detector.save")
        print(model.evaluate(test_x, test_y))
        print(model.metrics_names)


if __name__ == "__main__":
    cnn(mode="Train")
