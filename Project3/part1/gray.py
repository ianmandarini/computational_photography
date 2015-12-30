#================================================================================
#
# University of California, Berkeley
# CS194-26 (CS294-26): Computational Photography"
#
#================================================================================
#
# Project 3: "Fun with Frequencies"
#
# Student: Ian Albuquerque Raymundo da Silva"
# Email: ian.albuquerque@berkeley.edu
#
#================================================================================
#
# Special thanks to Alexei Alyosha Efros, Rachel Albert and Weilun Sun for help
# during lectures, office hours and questions on Piazza.
#
#================================================================================

#================================================================================
# Part 1: Hybrid Images (Gray Images)
#================================================================================

INPUT_IMAGES_NAME = ("venus.jpg","mona.jpg")

HIGH_FREQUENCY_BLUR_RATIO = 2
LOW_FREQUENCY_BLUR_RATIO = 4

#----------------------------------------------------------------------

from skimage import filter
from skimage import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
from align_image_code import align_images

def hybrid():

	# Read Images
	input_image_1 = skio.imread(INPUT_IMAGES_NAME[0])
	input_image_2 = skio.imread(INPUT_IMAGES_NAME[1])

#----------------------------------------------------------------------

	# Transform to float
	image_1 = img_as_float(input_image_1)
	image_2 = img_as_float(input_image_2)

#----------------------------------------------------------------------

	# Align images using provided functiom
	image_aligned_1, image_aligned_2 = align_images(image_1, image_2)

#----------------------------------------------------------------------

	# Divide image in color channels
	color_channels_1 = np.array([image_aligned_1[:,:,0],image_aligned_1[:,:,1],image_aligned_1[:,:,2]])
	color_channels_2 = np.array([image_aligned_2[:,:,0],image_aligned_2[:,:,1],image_aligned_2[:,:,2]])

	# Use color channels to compose an unique gray scale image
	# http://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
	image_1_grayscale = np.clip(color_channels_1[0]*0.299 + color_channels_1[1]*0.587 + color_channels_1[2]*0.114, 0.0, 1.0)
	image_2_grayscale = np.clip(color_channels_2[0]*0.299 + color_channels_2[1]*0.587 + color_channels_2[2]*0.114, 0.0, 1.0)

#----------------------------------------------------------------------

	# Apply the Laplacian Filter (high pass filter) to the first image by subtracting the Gaussian Filtered Image from the original
	low_frequency_image_1 = filter.gaussian_filter(image_1_grayscale, HIGH_FREQUENCY_BLUR_RATIO)
	high_frequency_image_1 = image_1_grayscale - low_frequency_image_1

	# Apply the Gaussian Filter (low pass filter) to the second image
	low_frequency_image_2 = filter.gaussian_filter(image_2_grayscale, LOW_FREQUENCY_BLUR_RATIO)

#----------------------------------------------------------------------

	# Add pictures together
	output_image = np.clip(high_frequency_image_1+low_frequency_image_2, 0.0, 1.0)

#----------------------------------------------------------------------

	# Show final image
	skio.imshow(output_image)
	skio.show()

#----------------------------------------------------------------------

if __name__ == '__main__':
	hybrid()
