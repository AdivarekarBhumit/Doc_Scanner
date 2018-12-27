import numpy as np 
import cv2
import imutils

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4,2), dtype='float32')
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def change_perspective(complete_image):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    img = cv2.resize(complete_image, (complete_image.shape[1]//2, complete_image.shape[0]//2))
#     ratio = img.shape[0] / 500.0
    ratio = 1.0
    orig = img.copy()
#     image = imutils.resize(img, height=500)
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # print("STEP 1: Edge Detection")
    # cv2.imshow("Image", image)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    
    # show the contour (outline) of the piece of paper
    # print("STEP 2: Find contours of paper")
    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Outline", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    #cv2.imwrite('Transformed.jpg', warped)

#     center = (warped.shape[1] // 2, warped.shape[0] // 2)
#     print("Center", center)
#     M = cv2.getRotationMatrix2D(center, 90, 1.0)
#     warped = cv2.warpAffine(warped, M, (warped.shape[1],warped.shape[0]))
    
    cv2.imwrite('rotated.jpg', cv2.resize(warped, (1280,720)))

    return cv2.resize(warped, (1280,720))

def remove_bg(image_name):
    image = cv2.imread(image_name)
    image_copy = np.copy(image)

    height, width, _ = image_copy.shape

    # Change color to RGB (from BGR) and then to HSV
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HSV)

    # Range of blue color
    lower_blue = np.array([100,25,0]) 
    upper_blue = np.array([160,255,255])

    # Define the masked area
    mask = cv2.inRange(image_copy, lower_blue, upper_blue)
    masked_image = np.copy(image_copy)
    masked_image[mask == 0] = [0, 0, 0]

    background_image = np.zeros_like(image)
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

    # Crop it to the right size
    crop_background = background_image[0:image.shape[0], 0:image.shape[1]]

    # Mask the cropped background so that the pizza area is blocked
    crop_background[mask == 0] = [0, 0, 0]
    # Add the two images together to create a complete image!
    complete_image = masked_image + crop_background

    # Now revert back to RGB
    complete_image = cv2.cvtColor(complete_image, cv2.COLOR_HSV2RGB_FULL)

    proper_image = change_perspective(complete_image)

remove_bg()