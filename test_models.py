import tensorflow as tf
import numpy as np
import cv2
from keras.models import model_from_json
from preprocess import label_dict

new_label_dict = {}
for key, value in label_dict.items():
    new_label_dict[value] = key

def load_model(model_architecture, model_weights):
    json_model = open(model_architecture, 'r')
    model = json_model.read()
    json_model.close()

    model = model_from_json(model)
    model.load_weights(model_weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def read_image(image):
    img = cv2.imread(image)
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.reshape(img, (-1,32,32,1))
    return img

model1 = load_model(model_architecture='./trained_models/model_basic.json', model_weights='./trained_models/model_basic.h5')
model2 = load_model(model_architecture='./trained_models/model_squeezenet.json', model_weights='./trained_models/model_squeezenet.h5')

print(model1.predict(read_image('./Passport/verdana_u_2_O.jpg')))
print(model2.predict(read_image('./Passport/verdana_u_2_O.jpg')))
