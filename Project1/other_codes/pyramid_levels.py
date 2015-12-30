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

import sys
import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.viewer.utils as utils
from skimage.transform import pyramid_gaussian
import math as math

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

if __name__ == "__main__":

	input_image_name = "../../data/emir.tif"

	input_image = skio.imread(input_image_name)

	image_height = np.floor(input_image.shape[0] / 3.0)

	input_image = sk.img_as_float(input_image)

	r = input_image[2*image_height: 3*image_height]
	g = input_image[image_height: 2*image_height]
	b = input_image[:image_height]

	pyramid_r = pyr(r)
	for i in range(0,len(pyramid_r)-1):

		skio.imshow(pyramid_r[i])
		skio.show()

	pyramid_g = pyr(g)
	for i in range(0,len(pyramid_g)-1):

		skio.imshow(pyramid_g[i])
		skio.show()

	pyramid_b = pyr(b)

	for i in range(0,len(pyramid_b)-1):

		skio.imshow(pyramid_b[i])
		skio.show()