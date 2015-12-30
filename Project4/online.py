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
# online.py
#
# Asks the user which operation will be performed on the image and computes
# that operation.
# 
#================================================================================

from skimage import img_as_float
import numpy as np
import skimage.io as skio
import seam_carving as sc

def run(input_file_name, input_order_file_name):

	input_image = skio.imread(input_file_name)
	input_order = open(input_order_file_name, 'r')

	image = img_as_float(input_image)

	# Read both orders (horizontal and vertical)
	shape_0 = float(input_order.readline())
	shape_1 = float(input_order.readline())
	order_h = np.zeros((shape_0,shape_1))
	for cell in np.nditer(order_h, op_flags=['readwrite'], order='C'):
		cell[...] = float(input_order.readline())

	shape_0 = float(input_order.readline())
	shape_1 = float(input_order.readline())
	order_v = np.zeros((shape_0,shape_1))
	for cell in np.nditer(order_v, op_flags=['readwrite'], order='C'):
		cell[...] = float(input_order.readline())

	while True:
		option = raw_input("Reduce image horizontally (h) or vertically (v)? ")
		if option == "h":
			reduce_by_h = int(raw_input("Choose reduce ratio horizontal (can be negative for beam insertion): "))
			output_image = sc.reduceImageHorizontally(image, order_h, reduce_by_h)

		elif option == "v":
			reduce_by_v = int(raw_input("Choose reduce ratio vertical (can be negative for beam insertion): "))
			output_image = sc.reduceImageHorizontally(sc.transposeImage(image), order_v.transpose(), reduce_by_v)
			output_image = sc.transposeImage(output_image)

		sc.display(output_image)