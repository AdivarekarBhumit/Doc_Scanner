import cv2
import numpy as np 

def get_brightness(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sum_brightness = np.sum(hsv[:,:,2])
    area = image.shape[0] * image.shape[1]

    avg = sum_brightness / area

    return avg

print(get_brightness(image_path='pan.png'))
print(get_brightness(image_path='dl.png'))
print(get_brightness(image_path='licence.jpg'))
print(get_brightness(image_path='pan2.png'))