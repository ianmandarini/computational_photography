#================================================================================
#
# University of California, Berkeley
# CS194-26 (CS294-26): Computational Photography
#
#================================================================================
#
# Project 7 - Part B: "Feature Matching for Autostitching"
#
# Student: Ian Albuquerque Raymundo da Silva
# Email: ian.albuquerque@berkeley.edu
#
#================================================================================
#
# Special thanks to Alexei Alyosha Efros, Rachel Albert and Weilun Sun for help
# during lectures, office hours and questions on Piazza.
#
#================================================================================

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, peak_local_max
import math
import random

import harris

import imageio
import manual
import transforms

#================================================================================

PLANAR_PANORAMA_FOLDER = "auto_inputs/planar"
CYLINDRICAL_PANORAMA_FOLDER = "auto_inputs/cylindrical"
PANORAMA_360_FOLDER = "inputs/360"

#================================================================================

HARRIS_TRESHOLD_RELATIVE = 0.1
DESCRIPTOR_SHAPE_XY = (8,8)
NUM_POINTS_TO_FIND = 1000
MIN_RADIUS_POINTS = 1
RUSSIAN_GRANDMA_TRESHOLD = 0.1
NUM_STEPS_RANSAC = 1000
INLINERS_TRESHOLD_RANSAC = 2

#================================================================================


def toGrayScale(img):
	channels = np.array([img[:,:,0],img[:,:,1],img[:,:,2]])
	gray = np.clip(channels[0]*0.299 + channels[1]*0.587 + channels[2]*0.114, 0.0, 1.0)
	return gray


def computeCornerResponse(img, radius, max_number_points,harris_threshold_rel):
	max_dimension = max([img.shape[0],img.shape[1]])

	gray_img = toGrayScale(img)

	edge_discard = 20
	harris_img, interest_points = harris.get_harris_corners(gray_img,edge_discard,harris_threshold_rel)

	num_points_found = len(interest_points[0])
	interest_points_transposed = np.transpose(interest_points)
	dists = harris.dist2(interest_points_transposed,interest_points_transposed)


	max_global = float("-inf")
	max_global_index = None
	for index in xrange(num_points_found):
		y = interest_points[0][index]
		x = interest_points[1][index]
		if harris_img[y,x] > max_global:
			max_global_index = index
			max_global = harris_img[y,x] 

	indexes_selected = [max_global_index]

	for r in reversed(xrange(radius,max_dimension)):
		for candidate_index in xrange(num_points_found):
			isGood = True
			for good_index in indexes_selected:
				if dists[candidate_index,good_index] < r*r:
					isGood = False
					break
			if isGood:
				indexes_selected.append(candidate_index)
				print "Found " + str(len(indexes_selected)) + " out of " + str(max_number_points) + " points expected."
				if len(indexes_selected) >= max_number_points:
					break
		if len(indexes_selected) >= max_number_points:
			break

	points = []
	for index in indexes_selected:
		points.append((interest_points[1][index],interest_points[0][index]))

	print "Total points used: " + str(len(points))

	figure, axis = plt.subplots(ncols=3)
	axis[0].imshow(img, vmin=0, vmax=1)
	axis[1].imshow(harris_img, vmin=0, vmax=1)
	axis[2].imshow(img, vmin=0, vmax=1)

	for x,y in points:
		marker=plt.Circle((x,y),2,color='r')
		axis[2].add_artist(marker)

	plt.show()

	return points

def extractDescriptors(img,points,shape):
	gray_img = toGrayScale(img)

	descriptors = []

	x_half = shape[0]/2.0
	y_half = shape[0]/2.0

	for x,y in points:
		descriptor = gray_img[y-y_half:y+y_half,x-x_half:x+x_half]
		descriptor = descriptor - np.min(descriptor)
		max_descriptor = np.max(descriptor)
		if max_descriptor != 0:
			descriptor = descriptor/max_descriptor
		descriptors.append(descriptor)

	return descriptors

def createMosaicDescriptors(descriptors, shape):
	num_descriptors = len(descriptors)

	mosaic_size = int(math.ceil(math.sqrt(num_descriptors)))

	mosaic = np.zeros((mosaic_size*shape[0],mosaic_size*shape[1]))

	for i in xrange(mosaic_size):
		for j in xrange(mosaic_size):
			if i*mosaic_size+j < num_descriptors:
				mosaic[i*shape[0]:(i+1)*shape[0],j*shape[1]:(j+1)*shape[1]] = descriptors[i*mosaic_size+j]

	return mosaic

def ssd(img_1,img_2):
	return np.sum((img_1-img_2)**2)

def defineCorrespondences(descriptors_1,points_1,descriptors_2,points_2,treshold):
	num_poins_1 = len(points_1)
	num_poins_2 = len(points_2)

	distance = np.zeros((num_poins_1,num_poins_2))

	for i1 in xrange(num_poins_1):
		for i2 in xrange(num_poins_2):
			distance[i1,i2] = ssd(descriptors_1[i1],descriptors_2[i2])

	points_selected_1 = []
	points_selected_2 = []

	for i1 in xrange(num_poins_1):

		best_distance = float("inf")
		best_index = None
		second_best_distance = float("inf")
		second_best_index = None

		for i2 in xrange(num_poins_2):
			d = distance[i1,i2]
			if d < best_distance:
				second_best_distance = best_distance
				second_best_index = best_index
				best_distance = d
				best_index = i2
			elif d < second_best_index:
				second_best_distance = d
				second_best_index = i2

		if best_distance/second_best_distance < treshold:
			points_selected_1.append(points_1[i1])
			points_selected_2.append(points_2[best_index])

	return points_selected_1, points_selected_2

def ssdPoint(pt1,pt2):
	return (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2

def ransac(points_1,points_2,transform_function,num_points,num_steps,inliners_margin):
	good_indexes = range(len(points_1))
	best_inliners = []


	for iterator in xrange(num_steps):
		n_indexes = random.sample(good_indexes,num_points)

		points_1_transform = [points_1[i] for i in n_indexes]
		points_2_transform = [points_2[i] for i in n_indexes]
		T = transform_function(points_1_transform,points_2_transform)

		inliners = []
		for index in good_indexes:
			point_it_should_be = transforms.applyHomogTransformation(T,points_1[index])
			point_it_is = points_2[index]
			distance = ssdPoint(point_it_is,point_it_should_be)
			if distance < inliners_margin:
				inliners.append(index)

		if len(inliners) > len(best_inliners):
			best_inliners = inliners

	good_points_1 = []
	good_points_2 = []
	for index in best_inliners:
		if points_1[index] not in good_points_1 and points_2[index] not in good_points_2:
			good_points_1.append(points_1[index])
			good_points_2.append(points_2[index])

	return good_points_1,good_points_2

def autoMerge(img_1, img_2, harris_threshold_rel, descriptor_shape, num_points_to_find, min_radius, russian_grandma_treshold, num_steps_ransac, inliners_treshold_ransac, transform_function, num_points_required_transform, mask = None):

	points_found_1 = computeCornerResponse(img_1,min_radius,num_points_to_find,harris_threshold_rel)
	descriptors_1 = extractDescriptors(img_1,points_found_1,descriptor_shape)

	#------------------------
	mosaic = createMosaicDescriptors(descriptors_1,descriptor_shape)

	figure, axis = plt.subplots(ncols=1)
	axis.imshow(np.dstack([mosaic,mosaic,mosaic]), vmin=0, vmax=1)
	plt.show()
	#------------------------

	points_found_2 = computeCornerResponse(img_2,min_radius,num_points_to_find,harris_threshold_rel)
	descriptors_2 = extractDescriptors(img_2,points_found_2,descriptor_shape)

	#------------------------
	mosaic = createMosaicDescriptors(descriptors_2,descriptor_shape)

	figure, axis = plt.subplots(ncols=1)
	axis.imshow(np.dstack([mosaic,mosaic,mosaic]), vmin=0, vmax=1)
	plt.show()
	#------------------------

	points_1, points_2 = defineCorrespondences(descriptors_1,points_found_1,descriptors_2,points_found_2,russian_grandma_treshold)


	#------------------------
	print "MATCH NUMBER OF POINTS:"
	print len(points_1)

	figure, axis = plt.subplots(ncols=2)
	axis[0].imshow(img_1, vmin=0, vmax=1)
	axis[1].imshow(img_2, vmin=0, vmax=1)

	for x,y in points_1:
		marker=plt.Circle((x,y),2,color='r')
		axis[0].add_artist(marker)
	for x,y in points_2:
		marker=plt.Circle((x,y),2,color='r')
		axis[1].add_artist(marker)

	plt.show()
	#------------------------

	print "Running RANSAC ... "

	ransaced_points_1, ransaced_points_2 = ransac(points_1,points_2,transform_function,num_points_required_transform,num_steps_ransac,inliners_treshold_ransac)


	#------------------------
	print "FINAL TOTAL OF POINTS:"
	print len(ransaced_points_1)

	figure, axis = plt.subplots(ncols=2)
	axis[0].imshow(img_1, vmin=0, vmax=1)
	axis[1].imshow(img_2, vmin=0, vmax=1)

	for x,y in ransaced_points_1:
		marker=plt.Circle((x,y),2,color='r')
		axis[0].add_artist(marker)
	for x,y in ransaced_points_2:
		marker=plt.Circle((x,y),2,color='r')
		axis[1].add_artist(marker)

	plt.show()

	#------------------------

	merged_image = manual.mergeImages(ransaced_points_1, ransaced_points_2, img_1, img_2, transform_function, mask)

	#------------------------
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(merged_image, vmin=0, vmax=1)
	plt.show()
	#------------------------

	return merged_image

def runAutoPanoramaPlane(folder_name):
	transform_function = transforms.computeHomography
	num_points_required_transform = 4
	mask = None

	file_names = imageio.getImageNames(folder_name)
	
	panorama = imageio.readImageFloat(file_names[0])
	for file_name in file_names[1:]:
		new_image = imageio.readImageFloat(file_name)
		panorama = autoMerge(new_image, panorama, HARRIS_TRESHOLD_RELATIVE, DESCRIPTOR_SHAPE_XY, NUM_POINTS_TO_FIND, MIN_RADIUS_POINTS, RUSSIAN_GRANDMA_TRESHOLD, NUM_STEPS_RANSAC, INLINERS_TRESHOLD_RANSAC, transform_function, num_points_required_transform, mask)

		print "Displaying the result so far:"
		figure, axis = plt.subplots(ncols=1)
		axis.imshow(panorama, vmin=0, vmax=1)
		plt.show()

	print "Displaying the final result."
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(panorama, vmin=0, vmax=1)
	plt.show()

def runAutoPanoramaCylinder(folder_name):
	transform_function = transforms.computeTranslation
	num_points_required_transform = 2

	file_names = imageio.getImageNames(folder_name)
	
	panorama,mask = manual.projectOnCylinder(imageio.readImageFloat(file_names[0]))

	for file_name in file_names[1:]:
		new_image,new_mask = manual.projectOnCylinder(imageio.readImageFloat(file_name))

		panorama = autoMerge(new_image, panorama, HARRIS_TRESHOLD_RELATIVE, DESCRIPTOR_SHAPE_XY, NUM_POINTS_TO_FIND, MIN_RADIUS_POINTS, RUSSIAN_GRANDMA_TRESHOLD, NUM_STEPS_RANSAC, INLINERS_TRESHOLD_RANSAC, transform_function, num_points_required_transform, new_mask)

		print "Displaying the result so far:"
		figure, axis = plt.subplots(ncols=1)
		axis.imshow(panorama, vmin=0, vmax=1)
		plt.show()

	print "Displaying the final result."
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(panorama, vmin=0, vmax=1)
	plt.show()

def runAutoPanoramaCylinderFull(folder_name):
	transform_function = transforms.computeTranslation
	num_points_required_transform = 2

	file_names = imageio.getImageNames(folder_name)
	
	panorama,mask = manual.projectOnCylinder(imageio.readImageFloat(file_names[0]))
	for file_name in file_names[1:]:
		new_image,new_mask = manual.projectOnCylinder(imageio.readImageFloat(file_name))

		panorama = autoMerge(new_image, panorama, HARRIS_TRESHOLD_RELATIVE, DESCRIPTOR_SHAPE_XY, NUM_POINTS_TO_FIND, MIN_RADIUS_POINTS, RUSSIAN_GRANDMA_TRESHOLD, NUM_STEPS_RANSAC, INLINERS_TRESHOLD_RANSAC, transform_function, num_points_required_transform, new_mask)
		
		print "Displaying the result so far:"
		figure, axis = plt.subplots(ncols=1)
		axis.imshow(panorama, vmin=0, vmax=1)
		plt.show()

	panorama_width = panorama.shape[1]

	points_1, points_2 = pointselecter.getPoints(source,reference)
	image360 = autoMerge(panorama, panorama, HARRIS_TRESHOLD_RELATIVE, DESCRIPTOR_SHAPE_XY, NUM_POINTS_TO_FIND, MIN_RADIUS_POINTS, RUSSIAN_GRANDMA_TRESHOLD, NUM_STEPS_RANSAC, INLINERS_TRESHOLD_RANSAC, transform_function, num_points_required_transform, new_mask)

	image360_width = image360.shape[1]

	image360 = image360[:,0.25*image360_width:0.25*image360_width+(panorama_width-(2*panorama_width-image360_width))]

	print "Displaying the final result."
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(image360, vmin=0, vmax=1)
	plt.show()

#================================================================================
# Runs the application!
if __name__ == '__main__':

	print "==============================================="
	print "==============================================="
	print "============== GO BEARS!!!!!!!! ==============="
	print "==============================================="
	print "==============================================="
	print ">> University of California, Berkeley"
	print ">> CS194-26 (CS294-26): Computational Photography"
	print "==============================================="
	print "==============================================="
	print ">> Project 7a: Image Warping and Mosaicing - Homographies"
	print ">> Student: Ian Albuquerque Raymundo da Silva"
	print ">> Email: ian.albuquerque@berkeley.edu"
	print "==============================================="
	print "==============================================="
	print "============== GO BEARS!!!!!!!! ==============="
	print "==============================================="
	print "==============================================="

	option = raw_input("Choose an option:\n1 = Planar Projection\n2 = Cylindrical Projection\nWrite Option Here: >> ")

	if option == "1":
		runAutoPanoramaPlane(PLANAR_PANORAMA_FOLDER)
	elif option == "2":
		runAutoPanoramaCylinder(CYLINDRICAL_PANORAMA_FOLDER)

	print "==============================================="
	print "==============================================="
	print "============== GO BEARS!!!!!!!! ==============="
	print "==============================================="
	print "==============================================="
	print ">> Have a good day!"
	print "==============================================="
	print "==============================================="
	print "============== GO BEARS!!!!!!!! ==============="
	print "==============================================="
	print "==============================================="