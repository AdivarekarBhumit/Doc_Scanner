import cv2

img = cv2.imread('test.jpg', 0)
gaus = cv2.GaussianBlur(img, (9,9), 10.0)
img = cv2.addWeighted(img, 1.5, gaus, -0.5, 0, img)
thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)


cv2.imshow('Thresh', thresh)
cv2.imwrite('thresh.jpg', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

