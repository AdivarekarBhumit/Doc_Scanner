import cv2
import numpy as np 
import statistics as st 

# Read Image
image = cv2.imread('./sample_images/pancard.jpg')
# Preprocess Image
image = cv2.resize(image, (1678, 1084))
img_copy = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.erode(image, (5,5), iterations=1)
blur = cv2.GaussianBlur(image, (11,11), 0)
th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
img = cv2.bitwise_not(th, th)
# Find Contours
image, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS)

count = 0
a = []
areas = []
width = []
height = []

for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    width.append(w)
    height.append(h)
    areas.append(cv2.contourArea(contour))

median = st.median(areas)
width = st.median(width)
height = st.median(height)

for contour in contours:
    [x, y, w, h] = cv2.boundingRect(contour)
    if h*w*10 > median and h in range(15, 90) and w in range(4, 85):
        count += 1
        a.append(contour)
        if w>width:
            cv2.rectangle(img_copy, (x,y), (x+int(width), y+h), (10,255,0), 2)
            w = int(w-width)
            x = int(x+width)
            cv2.rectangle(img_copy, (x,y), (x+w, y+h), (10,255,0), 2)
        else:
            cv2.rectangle(img_copy, (x,y), (x+w, y+h), (10,255,0), 2)

extracted_image = []
for contour in a:
        [x, y, w, h] = cv2.boundingRect(contour)
        new_image=img[y:y+h, x:x+w]
        extracted_image.append(new_image)

cv2.imwrite('segmented_image.jpg', img_copy)

for i in extracted_image:
    cv2.imshow('Extracted_Image', i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()