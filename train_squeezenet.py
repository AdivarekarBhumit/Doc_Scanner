import numpy as np 
import cv2 
from squeezenet import SqueezeNet
from preprocess import *
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from keras.utils import to_categorical

def main():
    pl_train, pl_labels = get_dataset('./Pan_Licence/')
    pl_labels = to_categorical(pl_labels, num_classes=36)

    x_train, x_val, y_train, y_val = train_test_split(pl_train, pl_labels, test_size=0.2, random_state=2064)

    tb = TensorBoard(log_dir='./logs/Squeezenet', write_graph=True)

    model = SqueezeNet()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.1, shuffle=True, callbacks=[tb])

    ## Save Model
    json_model = model.to_json()
    with open('model_squeezenet.json', 'w') as f:
        f.write(json_model)
    model.save_weights('model_squeezenet.h5')
    print('Model Saved')

    print('Evaluating Model')
    predict = model.evaluate(x=x_val, y=y_val, batch_size=1)

    print('Score',predict[1] * 100.00)
    print('Loss',predict[0])

if __name__ == "__main__":
    main()