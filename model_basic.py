import keras
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, Dense, MaxPool2D, Flatten
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from preprocess import *

def get_model():
    ## Create Model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', kernel_initializer='he_normal', input_shape=(32,32,1)))
    model.add(MaxPool2D((2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(36, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def main():
    pl_train, pl_labels = get_dataset('./Pan_Licence/')
    pl_labels = to_categorical(pl_labels, num_classes=36)

    x_train, x_val, y_train, y_val = train_test_split(pl_train, pl_labels, test_size=0.2, random_state=2064)

    tb = TensorBoard(log_dir='./logs', write_graph=True)

    model = get_model()

    print(model.summary())

    history = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_split=0.1, shuffle=True, callbacks=[tb])

    ## Save Model
    json_model = model.to_json()
    with open('model_basic.json', 'w') as f:
        f.write(json_model)
    model.save_weights('model_basic.h5')
    print('Model Saved')

    print('Evaluating Model')
    predict = model.evaluate(x=x_val, y=y_val, batch_size=1)

    print('Score',predict[1] * 100.00)
    print('Loss',predict[0])

if __name__ == "__main__":
    main()