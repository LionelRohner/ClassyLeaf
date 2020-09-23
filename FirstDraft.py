### ClassyLeaf

# Import Modules #

import cv2
import imutils
import numpy as np
from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from imutils import contours
from pylab import *
from skimage import measure

### Import Pics ###


print(cv2.__version__)

# set path and file
path = r'C:\Users\stryc\OneDrive\GitHub\Projects\ClassyLeaf\Pics'
# folder = '\F1'
filename = '\FA1.jpg'


# something was changed here...

# load image
image = cv2.imread(path + filename)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

### Downsampling ###

rescaled = rescale(image, 0.2, anti_aliasing=True)

### Normalize and threshold ###

# Gaussian blurring to reduce high frequency noise
blurred_img = cv2.GaussianBlur(rescaled, (7,7), 1.2)
mean_img, SD_img = cv2.meanStdDev(blurred_img)
min_img, max_img = np.amin(blurred_img), np.amax(blurred_img)

# threshold
thresh_img = cv2.threshold(blurred_img, max_img - 2*SD_img, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("Image", thresh_img)
cv2.waitKey(0)

labels = measure.label(thresh_img, background=0)
mask = np.zeros(thresh_img.shape, dtype="uint8")

# loop over the unique components
for label in np.unique(labels):

    # if this is the background label, ignore it
    if label == 0:
        continue

    # otherwise, construct the label mask and count the
    # number of pixels
    labelMask = np.zeros(thresh_img.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    print(numPixels)

    # if the number of pixels in the component is sufficiently
    # large, then add it to our mask of "large blobs"
    if numPixels > 10000:
        mask = cv2.add(mask, labelMask)
        # alternatively save masks individually and use biggest blob


blob_contour = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
blob_contour = imutils.grab_contours(blob_contour)

contours_img = cv2.drawContours(rescaled, blob_contour, -1, (0,255,0), 2)


cv2.imshow("Image", contours_img)
cv2.waitKey(0)


quit()



### Contour Detection ###