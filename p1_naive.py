##
#  p1_naive.py
#  Zhanwen Chen
#  Udacity Self-Driving Car Nanodegree
#  Project 1: Finding lanes
#  Naive solution to paint lane lines in a single image
#  using indexing: find indices of pixels that are both
#  above color thresholds and within region thresholds
#  and set these indices to a single color (in this case
#  to red, or [255, 0, 0] in rgb)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image

image = mpimg.imread("test_images/solidWhiteCurve.jpg")

# Grab the x and y sizes and
ysize = image.shape[0]
xsize = image.shape[1]
# make two copies of the image
color_select= np.copy(image)
line_image = np.copy(image)

# then we'll paint those pixels red in the original image to see our selection
# overlaid on the original.


## 1. Extract the pixels that meet our selection criteria,
# Define our color criteria
red_threshold = 150
green_threshold = 150
blue_threshold = 150
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define a triangle region of interest
left_bottom = [0, ysize]
right_bottom = [xsize, ysize]
apex = [xsize/2, ysize/2+35]

# Find lines between the apices by fitting two points into a 1-degree polynomial (linear fit)
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Use element-wise "or" to get the indices of pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
# Use the polynomial coefficients and constants to return
# the indices of pixels inside the triangle
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# Select the first image copy by the indices of pixels where
# the rgb values of the pixels are above the our defined
# rgb thresholds and set these pixels to black. In other
# words, we delete pixels below the color thresholds
color_select[color_thresholds] = [0,0,0]

# Find where image is both colored right and in the region.
# ~color_thresholds are the pixels color selected (or not deleted).
# region_thresholds are the pixels region selected
# For pixels both color selected and region selected,
# set them to red (255, 0, 0)
line_image[~color_thresholds & region_thresholds] = [255,0,0]

# Display our two output images
plt.imshow(line_image)
plt.show()

# save them to the test_images directory.
mpimg.imsave("test-after.jpg", line_image)
# im = Image.open("test-after.jpg")
# im.show()
