# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
# with tf.compat.v1.Session() as sess:
#     print(sess.run(c))

import numpy as np

# Importing necessary libraries
from skimage import data
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image

# Setting plot size to 15, 15
plt.figure(figsize=(15, 15))
img = Image.open('raw_images/20170607_142842.jpg')

# Sample Image of scikit-image package
coffee = data.coffee()
gray_coffee = rgb2gray(img)

# Computing Otsu's thresholding value
threshold = filters.threshold_otsu(gray_coffee)

# Computing binarized values using the obtained
# threshold
binarized_coffee = (gray_coffee > threshold) * 1
plt.subplot(2, 2, 1)
plt.title("Threshold: >" + str(threshold))

# Displaying the binarized image
plt.imshow(binarized_coffee, cmap="gray")

# Computing Ni black's local pixel
# threshold values for every pixel
threshold = filters.threshold_niblack(gray_coffee)

# Computing binarized values using the obtained
# threshold
binarized_coffee = (gray_coffee > threshold) * 1
plt.subplot(2, 2, 2)
plt.title("Niblack Thresholding")

# Displaying the binarized image
plt.imshow(binarized_coffee, cmap="gray")

# Computing Sauvola's local pixel threshold
# values for every pixel - Not Binarized
threshold = filters.threshold_sauvola(gray_coffee)
plt.subplot(2, 2, 3)
plt.title("Sauvola Thresholding")

# Displaying the local threshold values
plt.imshow(threshold, cmap="gray")

# Computing Sauvola's local pixel
# threshold values for every pixel - Binarized
binarized_coffee = (gray_coffee > threshold) * 1
plt.subplot(2, 2, 4)
plt.title("Sauvola Thresholding - Converting to 0's and 1's")

# Displaying the binarized image
plt.imshow(binarized_coffee, cmap="gray")
plt.show()