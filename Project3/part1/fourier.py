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
# Part 1: Hybrid Images (Fourier Transform Images)
#================================================================================

IMAGE_NAMES = ["aligned_venus.jpg","aligned_mona.jpg","vemona.jpg"]

#----------------------------------------------------------------------

from skimage import filter
from skimage import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
from align_image_code import align_images

#----------------------------------------------------------------------

def showFourier(input_image):

	# Read the image as floats
	image = img_as_float(input_image)

	# Generate the color channels
	color_channels = np.array([image[:,:,0],image[:,:,1],image[:,:,2]])

	f = []
	# Calculate the Fourie Transform for each channel to stack it
	for channel in color_channels:

		# Calculate the Fourier Transform
		fourier = np.log(np.abs(np.fft.fftshift(np.fft.fft2(channel))))

		# Adjust the image to fit in the [0,1] interval
		fourier = fourier - np.min(fourier)
		fourier = fourier / np.max(fourier)

		f.append(fourier)

	# Stack the channels to form a colored image
	fourier = np.dstack(f)

	# Show final result
	skio.imshow(fourier)
	skio.show()

#----------------------------------------------------------------------

def show():

	for image_name in IMAGE_NAMES:
		input_image = skio.imread(image_name)
		skio.imshow(input_image)
		skio.show()
		showFourier(input_image)

#----------------------------------------------------------------------

if __name__ == '__main__':
	show()