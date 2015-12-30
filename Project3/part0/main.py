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
# Part 0: Sharpening Images
#================================================================================

# The name of the image to be sharpened
INPUT_FILE_NAME = "cat.jpg"

# The size of the blur of the gaussian
BLUR_SIZE = 2

#----------------------------------------------------------------------

from skimage import filter
from skimage import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio

#----------------------------------------------------------------------

if __name__ == '__main__':
	# Reads the image
	input_image = skio.imread(INPUT_FILE_NAME)

	# Convert to float and separate in three different channels (R,G,B)
	image = img_as_float(input_image)
	color_channels = [image[:,:,0],image[:,:,1],image[:,:,2]]

#----------------------------------------------------------------------

	# For each channel:
	output_channels = []
	for channel in color_channels:

		# Create the Laplacian Filtered Image by subtracting the Gaussian Filtered Image
		blurred = filter.gaussian_filter(channel, BLUR_SIZE)
		highpass = channel - blurred

		# Add the Laplacian Filtered Image to the original image
		result = np.clip(channel+highpass, 0.0, 1.0)

		output_channels.append(result)

#----------------------------------------------------------------------

	# Colapse every channel into the final colored image
	output_image = np.dstack(output_channels)

#----------------------------------------------------------------------

	# Plot images (Before and After)
	fig, axes = plt.subplots(ncols=2)

	axes[0].imshow(image, vmin=0, vmax=1)
	axes[1].imshow(output_image, vmin=0, vmax=1)

	plt.show()