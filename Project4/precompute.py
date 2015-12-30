#================================================================================
#
# University of California, Berkeley
# CS194-26 (CS294-26): Computational Photography"
#
#================================================================================
#
# Project 4: "Seam Carving"
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
# File Description: 
#
# precompute.py
#
# Opens the image and energy mask and calculates the order of importance of each
# pixel in both horizontal and vertical deletions.
#
# Saves the ".ord" file.
#================================================================================

from skimage import img_as_float
import numpy as np
import skimage.io as skio
import seam_carving as sc

#----------------------------------------------------------------------

def run(input_file_name, output_order_file_name, use_energy_mask, energy_mask):

	input_image = skio.imread(input_file_name)

	image = img_as_float(input_image)

	if use_energy_mask:
		input_energy_mask = skio.imread(energy_mask)
		energy_mask = img_as_float(input_energy_mask)

		order_h = sc.computeHorizontalOrder(image, energy_mask)
		order_v = sc.computeHorizontalOrder(sc.transposeImage(image), sc.transposeImage(energy_mask))
		order_v = order_v.transpose()
	else:
		order_h = sc.computeHorizontalOrderNoMask(image)
		order_v = sc.computeHorizontalOrderNoMask(sc.transposeImage(image))
		order_v = order_v.transpose()

	output_file = open(output_order_file_name, 'w')
	sc.saveOrderFile(output_file, order_h)
	sc.saveOrderFile(output_file, order_v)
