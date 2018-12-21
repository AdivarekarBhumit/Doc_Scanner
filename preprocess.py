import numpy as np 
import cv2 
import os 

label_dict = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9,
                  'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'I':18,
                  'J':19, 'K':20, 'L':21, 'M':22, 'N':23, 'O':24, 'P':25, 'Q':26, 'R':27,
                  'S':28, 'T':29, 'U':30, 'V':31, 'W':32, 'X':33, 'Y':34, 'Z':35}

def get_label(img_name):
    name = img_name.split('.')[0]
    return label_dict[name[-1]]

def get_image_array(image):
    return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)

def get_dataset(path):
    labels = []
    images = []
    images_list = os.listdir(path)
    for img in images_list:
        images.append(get_image_array(path + img))
        labels.append(get_label(img))
    images = np.array(images).astype('float32')
    images = np.expand_dims(images, axis=3)
    labels = np.array(labels)
    images /= 255
    print(path[2:-1],'Train Array Size:',images.shape)
    return images, labels



