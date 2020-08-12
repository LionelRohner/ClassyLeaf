### ClassyLeaf

# Import Modules #

import cv2
import imutils
import numpy as np
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from imutils import contours

### Import Pics ###

# set path and file
path = r'C:\Users\stryc\OneDrive\GitHub\Projects\ClassyLeaf\Pics'
# folder = '\F1'
filename = '\FA1.jpg'

# load image
image = cv2.imread(path + filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cant get smaller
rescaled = rescale(image, 0.2, anti_aliasing=True)


### Downsampling ###



### Normalize and threshold ###

# Gaussian blurring to reduce high frequency noise
blurred_img = cv2.GaussianBlur(rescaled, (27,27), 1.2)
mean_img, SD_img = cv2.meanStdDev(blurred_img)
min_img, max_img = np.amin(blurred_img), np.amax(blurred_img)

thresh_img1 = cv2.threshold(blurred_img, max_img - SD_img, 255, cv2.THRESH_BINARY)[1]


cv2.imshow("Image", thresh_img1)
cv2.waitKey(0)


quit()








# thresholding of the images and conversion to a binary image
# 3SD away from max value of image (= 255) seems reasonable
thresh_img1 = cv2.threshold(blurred_img, max_img - 3 * SD_img, 255, cv2.THRESH_BINARY)[1]

# The next key step is to make the shape of the electrodes more clear. Need to flip the invert image again
# otherwise eroded and dilate are all backwards... I could not flip at the beginning and redo the normalization
# and take the lowest values but I m too lazy
thresh_img1 = cv2.bitwise_not(thresh_img1)

# create a nice ellipsoid kernel since we have round shape.
kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

### Contour Detection ###