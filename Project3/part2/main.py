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
# Part 2: Stacks
#================================================================================

# The name of the input image
INPUT_FILE_NAME = "vemona.jpg"

# The number of levels in the final stack
NUM_LEVELS = 6

#----------------------------------------------------------------------

from skimage import filter
from skimage import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio

#----------------------------------------------------------------------

if __name__ == '__main__':
		
	# Read the image
	input_image = skio.imread(INPUT_FILE_NAME)

	# Convert to float and separate in three different channels (R,G,B)
	image = img_as_float(input_image)
	color_channels = [image[:,:,0],image[:,:,1],image[:,:,2]]

	#----------------------------------------------------------------------

	# Generates the Gaussian Pyramid

	# For each level in the Pyramid
	gaussian_pyramids = []
	for level in range(NUM_LEVELS):

		# For each channel
		level_image_channels = []
		for channel in color_channels:

			# Blur the image with the Gaussian Filter (Low Pass Filter)
			# The blur needs to double for each level
			blurred_channel = filter.gaussian_filter(channel, 2**level)

			# Append the channel
			level_image_channels.append(blurred_channel)

		# Append the level
		gaussian_pyramids.append(level_image_channels)

	#----------------------------------------------------------------------

	# For each level, stack the three channels in the colored image
	output_gaussian_pyramid = []
	for level_image_channel in gaussian_pyramids:
		output_gaussian_pyramid.append(np.dstack(level_image_channel))

	#----------------------------------------------------------------------

	# Show each level of the Gaussian Picture
	for output_level in output_gaussian_pyramid:
		skio.imshow(output_level)
		skio.show()

	#----------------------------------------------------------------------

	# Generates the Laplacian Pyramid, given the Gaussian Pyramid

	# For each level in the Pyramid
	laplacian_pyramids = []
	for level in range(1,NUM_LEVELS):

		# For each channel
		level_image_channels = []
		for gaussian, gaussian_prev in zip(gaussian_pyramids[level], gaussian_pyramids[level-1]):

			# Generate the level by subtracting the current level in the gaussian pyramid by the last level
			sutraction = gaussian_prev - gaussian 

			# Adjust the image to fit in the interval [0,1]
			sutraction = sutraction - np.min(sutraction)
			sutraction = sutraction / np.max(sutraction)

			# Append the channel
			level_image_channels.append(sutraction)

		# Append the level
		laplacian_pyramids.append(level_image_channels)

	#----------------------------------------------------------------------

	# For the last level of the Laplacian Pyramid, add the last level of the Gaussian Pyramid
	laplacian_pyramids.append(gaussian_pyramids[NUM_LEVELS-1])

	#----------------------------------------------------------------------

	# For each level, stack the three channels in the colored image
	output_laplacian_pyramid = []
	for level_image_channel in laplacian_pyramids:
		output_laplacian_pyramid.append(np.dstack(level_image_channel))

	#----------------------------------------------------------------------

	# Show each level of the Laplacian Picture
	for output_level in output_laplacian_pyramid:
		skio.imshow(output_level)
		skio.show()