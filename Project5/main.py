#================================================================================
#
# University of California, Berkeley
# CS194-26 (CS294-26): Computational Photography"
#
#================================================================================
#
# Project 5: "Face Morphing"
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

import math
import sys
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skimage import img_as_float
from skimage.draw import polygon

#================================================================================

IMAGE_NAME_1 = "obama.jpg"
IMAGE_NAME_2 = "dilma.jpg"
POINTS_FILE = "obilma.pts"
NUM_FRAMES = 10

#================================================================================

# Opens the points file and return the points in it.
def openPointFile():
	points_file = open(POINTS_FILE, 'r')

	if not points_file:
		return None, None

	points_1 = []
	points_2 = []

	num_points = int(points_file.readline())
	for i in range(num_points):
		x = int(points_file.readline())
		y = int(points_file.readline())
		points_1.append((x,y))
		x = int(points_file.readline())
		y = int(points_file.readline())
		points_2.append((x,y))

	return points_1, points_2

#================================================================================

# Saves the points file.
def savePointFile(points_1,points_2):
	points_file = open(POINTS_FILE, 'w')
	points_file.write(str(len(points_1)) + "\n")
	for p1,p2 in zip(points_1,points_2):
		points_file.write(str(p1[0]) + "\n")
		points_file.write(str(p1[1]) + "\n")
		points_file.write(str(p2[0]) + "\n")
		points_file.write(str(p2[1]) + "\n")


#================================================================================

# Small interface for point input.
# Given two images, returns the corresponding points between them
def getPoints(image_1, image_2):

	# Get the dimensions
	image_1_height = image_1.shape[0]
	image_2_height = image_2.shape[0]
	image_1_width = image_1.shape[1]
	image_2_width = image_2.shape[1]

	# Used for controlling the alternate input between images
	click_on_image_1 = [True]

	# Show the images
	figure, axes = plt.subplots(ncols=2)
	axes[0].imshow(image_1, vmin=0, vmax=1)
	axes[1].imshow(image_2, vmin=0, vmax=1)

	# Use preset points?
	answer = raw_input("Used saved points from " + POINTS_FILE + "? (y/n)")
	if answer == 'y' or answer == 'Y':
		points_1, points_2 = openPointFile()
		for p in points_1:
			marker=plt.Circle(p,0.006*image_2_width,color='r')
			axes[0].add_artist(marker)	
		for p in points_2:
			marker=plt.Circle(p,0.006*image_2_width,color='r')
			axes[1].add_artist(marker)		
	else:
		points_1 = [(0,0),(image_1_width-1,0),(0,image_1_height-1),(image_1_width-1,image_1_height-1)]
		points_2 = [(0,0),(image_2_width-1,0),(0,image_2_height-1),(image_2_width-1,image_2_height-1)]
	print "Click on the images to assign points. Click on SPACE to finish point selection."


	# Show the triangulation for the given points
	lines = [[],[]]

	for l1,l2 in zip(lines[0],lines[1]):
		axes[0].lines.remove(l1)
		axes[1].lines.remove(l2)

	lines[0] = []
	lines[1] = []

	points_inter = (np.array(points_1) + np.array(points_2)) / 2.0

	delanayTriangulation = Delaunay(points_inter)
	triangles = delanayTriangulation.simplices

	edges = []
	for t in triangles:
		edges.append((t[0],t[1]))
		edges.append((t[1],t[2]))
		edges.append((t[2],t[0]))

	for e in edges:
		xdata = [points_1[e[0]][0],points_1[e[1]][0]]
		ydata = [points_1[e[0]][1],points_1[e[1]][1]]
		l1 = plt.Line2D(xdata,ydata)
		lines[0].append(l1)
		axes[0].add_line(l1)
		xdata = [points_2[e[0]][0],points_2[e[1]][0]]
		ydata = [points_2[e[0]][1],points_2[e[1]][1]]
		l2 = plt.Line2D(xdata,ydata)
		lines[1].append(l2)
		axes[1].add_line(l2)

	# Event for presing space
	def onKeyPress(event):
		if event.key == ' ':
			if click_on_image_1[0]:
				plt.close()
			else:
				print "You need to assign a corresponding point in image 2."

	# Event for clicking on the images
	def clickOnImage(event):
		if event.button == 1:
			# Image one selected
			if event.inaxes == axes[0]:
				if click_on_image_1[0]:
					print "Point in image 1 selected."
					points_1.append((int(event.xdata),int(event.ydata)))
					marker=plt.Circle((event.xdata,event.ydata),0.006*image_1_width,color='r')
					axes[0].add_artist(marker)
					click_on_image_1[0] = False
				else:
					print "Please, select a point in image 2."
			# Image two selected
			if event.inaxes == axes[1]:
				if click_on_image_1[0]:
					print "Please, select a point in image 1."
				else:
					print "Point in image 2 selected."
					points_2.append((int(event.xdata),int(event.ydata)))
					marker=plt.Circle((event.xdata,event.ydata),0.006*image_2_width,color='r')
					axes[1].add_artist(marker)
					click_on_image_1[0] = True
					
					plt.show(block=False)
					plt.draw()

					# Update the triangulation for the given points
					# Slow, but good for helping the user
					for l1,l2 in zip(lines[0],lines[1]):
						axes[0].lines.remove(l1)
						axes[1].lines.remove(l2)

					lines[0] = []
					lines[1] = []

					points_inter = (np.array(points_1) + np.array(points_2)) / 2.0

					delanayTriangulation = Delaunay(points_inter)
					triangles = delanayTriangulation.simplices

					edges = []
					for t in triangles:
						edges.append((t[0],t[1]))
						edges.append((t[1],t[2]))
						edges.append((t[2],t[0]))

					for e in edges:
						xdata = [points_1[e[0]][0],points_1[e[1]][0]]
						ydata = [points_1[e[0]][1],points_1[e[1]][1]]
						l1 = plt.Line2D(xdata,ydata)
				 		lines[0].append(l1)
				 		axes[0].add_line(l1)
						xdata = [points_2[e[0]][0],points_2[e[1]][0]]
						ydata = [points_2[e[0]][1],points_2[e[1]][1]]
						l2 = plt.Line2D(xdata,ydata)
				 		lines[1].append(l2)
				 		axes[1].add_line(l2)

		plt.show(block=False)
		plt.draw()

	# Set up the events
	listener_button_press = figure.canvas.mpl_connect('button_press_event', clickOnImage)
	listener_key_press = figure.canvas.mpl_connect('key_press_event', onKeyPress)

	# Show imaves
	plt.show()
	plt.draw()

	# Close everything
	figure.canvas.mpl_disconnect(listener_button_press)
	figure.canvas.mpl_disconnect(listener_key_press)

	plt.close()

	# Check if points are OK
	if len(points_1) != len(points_2):
		exit("The number of points selected in both images is not the same!")

	# Save points?
	answer = raw_input("Save points file on " + POINTS_FILE + "? (y/n)")
	if answer == 'y' or answer == 'Y':
		savePointFile(points_1,points_2)

	return np.array(points_1), np.array(points_2)

#================================================================================

# Checks if two images have the same size
def checkForImageSize(image_1,image_2):
	if image_1.shape != image_2.shape:
		exit("Images do not have the same size.")

#================================================================================	

# Compute the (inverse) affine transformation of one triangle from one point array
# to the corresponding one. (Basically, a lot of linear algebra with
# homegeneous coordinates)
def computeAffine(t,points_1,points_2):
	A_1 = np.array(points_1[t[0]])
	B_1 = np.array(points_1[t[1]])
	C_1 = np.array(points_1[t[2]])

	A_2 = np.array(points_2[t[0]])
	B_2 = np.array(points_2[t[1]])
	C_2 = np.array(points_2[t[2]])

	p_1 = B_1 - A_1
	q_1 = C_1 - A_1

	p_2 = B_2 - A_2
	q_2 = C_2 - A_2

	T_1 = np.matrix([[p_1[0],q_1[0],A_1[0]],[p_1[1],q_1[1],A_1[1]],[0,0,1]])
	T_1_inv = T_1.getI()
	T_2 = np.matrix([[p_2[0],q_2[0],A_2[0]],[p_2[1],q_2[1],A_2[1]],[0,0,1]])
	T_2_inv = T_2.getI()

	T = np.dot(T_1,T_2_inv)

	return T

#================================================================================

# Do the morph of the origin image, with its origin points to the destination points
# using the triangulation described in triangles
def morph(origin_image,origin_points,destination_points,triangles):
	
	output_image = origin_image*0
 
 	# For each triangle
	for triangle in triangles:
		# Computes the (inverse) affine
		T = computeAffine(triangle,origin_points,destination_points)

		# Get points of the triangle
		img = np.zeros((origin_image.shape[0], origin_image.shape[1]), dtype=np.uint8)
		x = np.array([destination_points[triangle[0]][0], destination_points[triangle[1]][0], destination_points[triangle[2]][0]])
		y = np.array([destination_points[triangle[0]][1], destination_points[triangle[1]][1], destination_points[triangle[2]][1]])
		rr, cc = polygon(y, x)
		img[rr, cc] = 1
		triangle_pixels = np.where(img == 1)

		# Fill the output images with the correct pixels
		for y,x in zip(triangle_pixels[0],triangle_pixels[1]):
			hom = np.array([x,y,1.0])
			origin_point_hom = np.array(np.dot(T,hom))[0]
			output_image[y,x] = origin_image[int(origin_point_hom[1]),int(origin_point_hom[0])]

	return output_image

#================================================================================

if __name__ == '__main__':

	# Reads input
	input_image_1 = skio.imread(IMAGE_NAME_1)
	input_image_2 = skio.imread(IMAGE_NAME_2)

	checkForImageSize(input_image_1,input_image_2)

	# Get the points
	points_1, points_2 = getPoints(input_image_1, input_image_2)	

	# Calculates midway points
	points_inter = (points_1 + points_2) / 2.0

	# Calculate triangulation for that
	delanayTriangulation = Delaunay(points_inter)
	triangles = delanayTriangulation.simplices

	# Calculates the midway face
	alpha = 0.5
	points_inter = (1.0-alpha)*points_1 + alpha*points_2

	print "Calculating midway face."

	image_1_morphed = np.array(morph(input_image_1,points_1,points_inter,triangles))
	image_2_morphed = np.array(morph(input_image_2,points_2,points_inter,triangles))

	image_1_morphed = img_as_float(image_1_morphed)
	image_2_morphed = img_as_float(image_2_morphed)

	print "Showing midway face."

	output_image = (1.0-alpha)*image_1_morphed + alpha*image_2_morphed
	figure, axes = plt.subplots(ncols=3)
	axes[0].imshow(image_1_morphed, vmin=0, vmax=1)
	axes[1].imshow(image_2_morphed, vmin=0, vmax=1)
	axes[2].imshow(output_image, vmin=0, vmax=1)
	plt.show()
	plt.draw()
	plt.close()

	# Iterate through frames, calculating the morphed faces
	alpha = 0
	N = NUM_FRAMES
	d = 1.0/float(N)

	print "Calculating " + str(N) + " frames for the morph."
	for i in range(N+1):
		alpha = i*d
		points_inter = (1.0-alpha)*points_1 + alpha*points_2

		print "Calculating frame alpha="+str(alpha)+"."

		image_1_morphed = np.array(morph(input_image_1,points_1,points_inter,triangles))
		image_2_morphed = np.array(morph(input_image_2,points_2,points_inter,triangles))

		image_1_morphed = img_as_float(image_1_morphed)
		image_2_morphed = img_as_float(image_2_morphed)

		output_image = (1.0-alpha)*image_1_morphed + alpha*image_2_morphed

		print "Showing frame alpha="+str(alpha)+"."

		figure, axes = plt.subplots(ncols=3)
		axes[0].imshow(image_1_morphed, vmin=0, vmax=1)
		axes[1].imshow(image_2_morphed, vmin=0, vmax=1)
		axes[2].imshow(output_image, vmin=0, vmax=1)
		plt.show()
