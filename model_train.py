from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from audio_classifier import AudioClassifier
from tensorflow.keras.utils import to_categorical

train=np.load("/tmp/X_train.npz")
test=np.load("/tmp/X_test.npz")
val=np.load("/tmp/X_val.npz")

X_train=train['x']
y_train=to_categorical(train['y'])
X_test=test['x']
y_test=to_categorical(test['y'])
X_val=val['x']
y_val=to_categorical(val['y'])

model=AudioClassifier()
model.summary()

model.compile(optimizer='adam',
              loss="categorical_crossentrophy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=128, validation_data=(X_val, y_val))






