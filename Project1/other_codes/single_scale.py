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

import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.viewer.utils as utils
from skimage.transform import pyramid_gaussian

#================================================================================

def crop_borders(img,percentage):
	x_crop = percentage*img.shape[0]/2
	y_crop = percentage*img.shape[1]/2

	return img[x_crop:img.shape[0]-x_crop,y_crop:img.shape[1]-y_crop]

def img_translate(img,dx,dy):
	result = np.roll(img, dx, axis = 1)
	result = np.roll(result, dy, axis = 0)
	return result

def sum_of_squared_diff(img_1, img_2):
	return np.sum((img_1 - img_2)**2)

#================================================================================

def align(_img_1, _img_2, metric):
	best_value = float("inf")

	best_dx = 0
	best_dy = 0

	img_1 = crop_borders(_img_1,0.1)
	img_2 = crop_borders(_img_2,0.1)

	for dx in range(-15,16):
		for dy in range(-15,16):

			ssd = metric(img_translate(img_1,dx,dy),img_2)

			if ssd < best_value:
				best_value = ssd
				best_dx = dx
				best_dy = dy

	print "Alignment:"
	print "dx = " + str(best_dx)
	print "dy = " + str(best_dy)

	return img_translate(_img_1,best_dx,best_dy)

#================================================================================

if __name__ == "__main__":

	imname = "../../data/cathedral.jpg"

	im = skio.imread(imname)
	height = np.floor(im.shape[0] / 3.0)

	im = sk.img_as_float(im)

	b = im[:height]
	g = im[height: 2*height]
	r = im[2*height: 3*height]

	ar = align(r, g, sum_of_squared_diff)
	ag = g
	ab = align(b, g, sum_of_squared_diff)

	ar = crop_borders(ar,0.1)
	ag = crop_borders(ag,0.1)
	ab = crop_borders(ab,0.1)

	im_out = np.dstack([ar, ag, ab])

	skio.imshow(im_out)
	skio.show()