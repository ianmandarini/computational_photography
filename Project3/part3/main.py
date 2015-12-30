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
# Part 3: Blended Images
#================================================================================

# The names of the input images
INPUT_IMAGE_NAME_1 = "lion.jpg"
INPUT_IMAGE_NAME_2 = "mico.jpg"
MASK_NAME = "milion_mask.jpg"

# Number of levels to blend
BLENDING_LEVELS = 6

#----------------------------------------------------------------------

from skimage import filter
from skimage import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio

#----------------------------------------------------------------------

# Generates the Gaussian Pyramid
def generateGaussianPyramids(image, levels):
	color_channels = [image[:,:,0],image[:,:,1],image[:,:,2]]

	gaussian_pyramids = []
	for level in range(levels):
		level_image_channels = []
		for channel in color_channels:
			blurred_channel = filter.gaussian_filter(channel, 2**level)
			level_image_channels.append(blurred_channel)
		gaussian_pyramids.append(level_image_channels)
	return gaussian_pyramids

# Generates the Laplacian Pyramid
def generateLaplacianPyramids(image, levels):
	gaussian_pyramids = generateGaussianPyramids(image,levels)

	laplacian_pyramids = []
	for level in range(1,levels):
		level_image_channels = []
		for gaussian, gaussian_prev in zip(gaussian_pyramids[level], gaussian_pyramids[level-1]):
			sutraction = gaussian_prev - gaussian
			level_image_channels.append(sutraction)
		laplacian_pyramids.append(level_image_channels)
	laplacian_pyramids.append(gaussian_pyramids[levels-1])
	return laplacian_pyramids	

#----------------------------------------------------------------------

if __name__ == '__main__':

	# Read Images
	input_image_1 = skio.imread(INPUT_IMAGE_NAME_1)
	input_image_2 = skio.imread(INPUT_IMAGE_NAME_2)
	input_mask = skio.imread(MASK_NAME)

	# Convert images to float
	image1 = img_as_float(input_image_1)
	image2 = img_as_float(input_image_2)
	mask = img_as_float(input_mask)

#----------------------------------------------------------------------

	# Generate the Laplacian Pyramid for the input images
	# Generate the Gaussian Pyramid for the mask
	laplacian_pyramid_1 = generateLaplacianPyramids(image1,BLENDING_LEVELS)
	laplacian_pyramid_2 = generateLaplacianPyramids(image2,BLENDING_LEVELS)
	gaussian_mask = generateGaussianPyramids(mask,BLENDING_LEVELS)

#----------------------------------------------------------------------
	# For each level in the pyramids
	result_pyramid = []
	for im1, im2, msk in zip(laplacian_pyramid_1,laplacian_pyramid_2,gaussian_mask):

		# For each color channel
		result_level_channels = []
		for ch1, ch2, chm in zip(im1, im2, msk):

			# Add the images using the mask as weight
			ch1_processed = ch1 * chm
			ch2_processed = ch2 * (1 - chm)
			ch_result = ch1_processed + ch2_processed

			# Append the level
			result_level_channels.append(ch_result)

		# Colapse the channels into the colored image
		result_level = np.dstack(result_level_channels)

		# Add the final level to the pyramid
		result_pyramid.append(result_level)

#----------------------------------------------------------------------

	# Iterates through the levels adding everything together
	output_image = result_pyramid[0] - result_pyramid[0]
	for img in result_pyramid:
		output_image = output_image + img

	# Clips the image
	output_image = np.clip(output_image,0.0,1.0)

#----------------------------------------------------------------------

	# Show input images and output
	skio.imshow(image1)
	skio.show()
	skio.imshow(mask)
	skio.show()
	skio.imshow(image2)
	skio.show()
	skio.imshow(output_image)
	skio.show()





