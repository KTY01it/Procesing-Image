import cv2 as cv
import numpy as np
import imutils
from matplotlib import pyplot as plt
from flatten_json import flatten

img_path = 'C:/Users/ADMIN/Downloads/15/picture29.jpg'

img = cv.imread(img_path)
# cv.imshow('image', img)
# cv.waitKey(0)

img_gray = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

rotate = imutils.rotate(img_gray, -3)
# print(rotate.shape) # Print image shape

new_width = 800
new_height = 600
img_resized = cv.resize(src=rotate, dsize=(new_width, new_height))
# reisze_img_name = 'test1_%dx%d.jpg' % (new_width, new_height)
# cv.imwrite(reisze_img_name, img_resized)
# print(reisze_img_name, img_resized.shape)


cropped_image = img_resized[190:250, 280:520]
# cv.imshow('image_his1', cropped_image)


# original = cropped_image.copy()
# xp = [0, 64, 128, 192, 255]
# fp = [0, 16, 128, 240, 255]
# x = np.arange(256)
# label = np.interp(x, xp, fp).astype('uint8')
# img1 = cv.LUT(img_resized, label)

# gauss = cv.GaussianBlur(cropped_image, (7,7), 0)
# unsharp_image = cv.addWeighted(cropped_image, 2, gauss, -1, 0)

# cv.imshow('image_his', img1)

# Adjust the brightness and contrast
# Adjusts the brightness by adding 10 to each pixel value
brightness = 10 
# Adjusts the contrast by scaling the pixel values by 2.3
contrast = 2.3 
image2 = cv.addWeighted(cropped_image, contrast, np.zeros(cropped_image.shape, img_resized.dtype), 0, brightness)
cv.imshow('image_his3', image2)

kernel2 = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
image_sharp = cv.filter2D(src=image2, ddepth=-1, kernel=kernel2)
cv.imshow('image_his5', image_sharp)


# # cropped_image2 = image2[40:90, 80:115]
# cropped_image2 = image2[0:51, 100:135]

# cropped_image2 = image2[0:51, 150:185]

cropped_image2 = image_sharp[0:51, 190:230]

cv.imshow('cropped_image2', cropped_image2)

img_blur = cv.blur(src=cropped_image2, ksize=(3,3)) # Using the blur function to blur an image where ksize is the kernel size

# kernel = np.ones((5, 5), np.float32)/25
# dst = cv.filter2D(cropped_image2, -1, kernel)

hist,bins = np.histogram(img_blur.flatten(), 256, [0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max())/cdf.max()
# plt.plot(cdf_normalized, color='b')
# plt.hist(img_blur.flatten(),256, [0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min())*255 / (cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m, 0) .astype('uint8')

img_his = cdf[img_blur]

# # resize_img_name1 = 'picture.jpg'
# # cv.imwrite(resize_img_name1, img_his)
thresh, img_binary = cv.threshold(img_his, thresh=90, maxval=255, type=cv.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
img_erosion = cv.erode(img_binary, kernel, iterations=1)
# img_dilation = cv.dilate(img_erosion, kernel, iterations=1)

# cv.imshow('image_his', cropped_image2)

# edged = cv.Canny(cropped_image2, 30, 100)
# cv.imshow('image_his0', edged)


# contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
# img_new = cv.drawContours(cropped_image2, contours, -1, (0, 255, 0), 3)

cv.imshow('image_his1', img_his)
cv.imshow('img_binary', img_binary)
cv.imshow('img_erosion', img_erosion)
# cv.imshow('img_dilation', img_dilation)
cv.waitKey(0)
cv.destroyAllWindows()