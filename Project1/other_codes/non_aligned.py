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

#================================================================================

if __name__ == "__main__":
	
	input_image_name = "../../data/emir.tif"

	input_image = skio.imread(input_image_name)
	image_height = np.floor(input_image.shape[0] / 3.0)

	input_image = sk.img_as_float(input_image)

	r = input_image[2*image_height: 3*image_height]
	g = input_image[image_height: 2*image_height]
	b = input_image[:image_height]

	output_image = np.dstack([r, g, b])

	skio.imshow(output_image)
	skio.show()