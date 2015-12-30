#================================================================================
# University of California, Berkeley
# CS194-26 (CS294-26): Computational Photography"
#
#================================================================================
#
# Project 1: Images of the Russian Empire"
#
# Student: Ian Albuquerque Raymundo da Silva"
# Email: ian.albuquerque@berkeley.edu
#
#================================================================================
# Reference used for creating this code:
#
# CS194-26 (CS294-26) - Project 1 Starter Python code - Provided with the Assignment
# Avaiable on: http://inst.eecs.berkeley.edu/~cs194-26/fa15/hw/proj1/data/colorize_skel.py
#
# "A crash course on NumPy for images"
# Avaiable on: http://scikit-image.org/docs/dev/user_guide/numpy_images.html
#
# "Image manipulation and processing using Numpy and Scipy"
# Avaiable on: https://scipy-lectures.github.io/advanced/image_processing/
#
# "Build image pyramids"
# Avaiable on: http://scikit-image.org/docs/dev/auto_examples/plot_pyramid.html
#
# "Module: Transform"
# Avaiable on: http://scikit-image.org/docs/dev/api/skimage.transform.html
#
#================================================================================
#
# Special thanks to Alexei Alyosha Efros, Rachel Albert and Weilun Sun for help
# during lectures, office hours and questions on Piazza.
#
#================================================================================

#================================================================================
# Modules imports:
#================================================================================

import sys
import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.viewer.utils as utils
import skimage.transform
import math

#================================================================================
# Useful Functions
#================================================================================

# Crops the borders of a given images by a constant factor given "percentage"
def crop_borders(img, percentage):

	x_crop = percentage*img.shape[0]/2
	y_crop = percentage*img.shape[1]/2

	return img[x_crop:img.shape[0]-x_crop,y_crop:img.shape[1]-y_crop]

# Translates an image by (dx,dy)
def img_translate(img,dx,dy):

	result = np.roll(img, dx, axis = 1)
	result = np.roll(result, dy, axis = 0)

	return result

# Calculate the Sum of Squared Differences of two given images
def sum_of_squared_diff(img_1, img_2):
	return np.sum((img_1 - img_2)**2)

#================================================================================
# Single Scale Alignment
#================================================================================

# Given two images, find the best alignment for image 1 compared to image 2, given
# a metric to compare two images (In this case, the SSD)
def best_alignment(img_1, img_2, metric):

	# Best alignments so far
	best_value = float("inf")
	best_dx = 0
	best_dy = 0

	# Defines a search window of displacements of [-10,10] for each direction.
	# For images that do not have more than 10 pixels of width and height, stick
	# lower the window to their resolution
	width_search_window = min(10,img_1.shape[0]/2)
	height_search_window = min(10,img_1.shape[1]/2)

	# Iterates through all possibilites of displacement using the defined window
	for dx in range(-width_search_window,width_search_window):
		for dy in range(-height_search_window,height_search_window):

			# Calculate the metric given
			ssd = metric(img_translate(img_1,dx,dy),img_2)

			# If the result is better, stick with it
			if ssd < best_value:
				best_value = ssd
				best_dx = dx
				best_dy = dy

	return (best_dx, best_dy)

#================================================================================
# Multi Scale Alignment
#================================================================================

# Given two image pyramids, find the best alignment for image 1 compared to image 2, given
# a metric to compare two images (In this case, the SSD)

def align_pyramids(_pyramid_img_1, pyramid_img_2, metric):

	# First, copy the first image pyramid to be sure we will not overwrite any input data.
	pyramid_img_1 = _pyramid_img_1

	# Best alignments so far
	best_dx, best_dy = 0, 0

	# Level by level, try to align the images using the single scale alignment
	for image_1, image_2 in zip(reversed(pyramid_img_1), reversed(pyramid_img_2)):

		# Whatever was the best alignment received from the last level, double it
		# This must be done because the amount of pixels in the new levels will be larger
		# For each pixel in one level, there are 4 pixels in the next level
		# This factor of 2 takes that in consideration for the aligment
		best_dx *= 2
		best_dy *= 2

		# Translates the current image with the best alignment found so far
		image_1 = img_translate(image_1,best_dx,best_dy)

		# Crops the image to make sure we get rid of the noisy borders
		image_1 = crop_borders(image_1,0.2)
		image_2 = crop_borders(image_2,0.2)

		# Try to align the image with the single scale alignment
		new_dx, new_dy = best_alignment(image_1,image_2,metric)

		# Add the results received to the alignment found so far
		best_dx += new_dx
		best_dy += new_dy

	# Last but not least, translates the original image using the best alignment found
	return (img_translate(pyramid_img_1[0],best_dx,best_dy),[best_dx,best_dy])

#================================================================================
# Pyramid Creation Function
#================================================================================

# Given a image, creates its image pyramid
def pyr(img):

	# Calculate in how many levels each dimension can be divided into
	width_dim = int(math.log(img.shape[0],2))
	height_dim = int(math.log(img.shape[1],2))

	# The number of levels in the image will be the minimum of the two
	# values calculated above
	num_images = min(width_dim,height_dim)

	# Initializes the image array
	pyramid = []

	# Adds the original image as the 'level 0' image
	pyramid.append(np.asarray(img))

	# Generate each level by using a rescaled version of the previous level
	# The scaling factor is 2
	# Information is lost on every level
	for i in range(1,num_images+1):
		pyramid.append(sk.transform.rescale(pyramid[i-1],0.5))

	return pyramid

#================================================================================
# A Wrapping Function for Aligning the three Channels Given
#================================================================================

# Align the red, green and blue channels comparing them to the green image
def align_channels(r, g, b):

	# Defines the metric to use
	metric = sum_of_squared_diff

	# Create the pyramids
	print "Creating red image pyramid..."
	pyramid_r = pyr(r)
	print "Creating green image pyramid..."
	pyramid_g = pyr(g)
	print "Creating blue image pyramid..."
	pyramid_b = pyr(b)

	# Find the best alignments based on the pyramids
	print "Aligning the red image compared to the green image (this might take a while)..."
	result_r, r_d = align_pyramids(pyramid_r, pyramid_g, metric)
	result_g, g_d = pyramid_g[0], [0, 0]
	print "Aligning the blue image compared to the green image (this might take a while)..."
	result_b, b_d = align_pyramids(pyramid_b, pyramid_g, metric)

	return (result_r, result_g, result_b, r_d, g_d, b_d)

#================================================================================

if __name__ == "__main__":

	print "###############################################"
	print "CS194-26 (CS294-26): Computational Photography"
	print "###############################################"
	print "Project 1: Images of the Russian Empire"
	print "Student: Ian Albuquerque Raymundo da Silva"
	print "###############################################"

	input_image_path = "../data/emir.tif"

	print "Reading 'emir.tif' data file from '../data/emir.tif' ..."

	input_image = skio.imread(input_image_path)

	image_height = np.floor(input_image.shape[0] / 3.0)
	input_image = sk.img_as_float(input_image)

	r = input_image[2*image_height: 3*image_height]
	g = input_image[image_height: 2*image_height]
	b = input_image[:image_height]

	print "Starting aligning process..."

	aligned_r, aligned_g, aligned_b, r_d, g_d, b_d = align_channels(r,g,b)

	print "Alignment Complete !!! Results:"
	print "R Alignment = (" + str(r_d[0]) + "," + str(r_d[1]) + ")"
	print "G Alignment = (" + str(g_d[0]) + "," + str(g_d[1]) + ") [Reference]"
	print "B Alignment = (" + str(b_d[0]) + "," + str(b_d[1]) + ")"

	print "Cropping borders by 10%..."

	aligned_r = crop_borders(aligned_r, 0.1)
	aligned_g = crop_borders(aligned_g, 0.1)
	aligned_b = crop_borders(aligned_b, 0.1)

	print "Stacking processed images..."
	output_image_aligned = np.dstack([aligned_r, aligned_g, aligned_b])

	print "Displaying output image..."
	skio.imshow(output_image_aligned)

	print "DONE!"
	skio.show()