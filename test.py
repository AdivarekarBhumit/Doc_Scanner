import cv2
from preprocess import *

passport_imgs, p_labels = get_dataset('./Passport/')
pan_licence_imgs, pl_labels = get_dataset('./Pan_Licence/')


cv2.imshow('Image', passport_imgs[5])
print(p_labels[5])
cv2.waitKey(0)
cv2.destroyAllWindows()