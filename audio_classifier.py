from tensorflow import keras
from tensorflow.keras import layers

def AudioClassifier():

    inputs=keras.Input(shape=(64,688,2))
    x = layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(rate=0.2)(x)

    x = layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(rate=0.2)(x)

    x = layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    x = layers.Dropout(rate=0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(rate=0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model





