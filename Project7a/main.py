#================================================================================
#
# University of California, Berkeley
# CS194-26 (CS294-26): Computational Photography
#
#================================================================================
#
# Project 7 - Part A: "Image Warping and Mosaicing - Homographies"
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

#================================================================================
# CONSTANTS AND SETTINGS
#================================================================================

RECTIFICATION_FILE_NAME = "inputs/rectification/times_square.jpg"

PLANAR_PANORAMA_FOLDER = "inputs/planar"
CYLINDRICAL_PANORAMA_FOLDER = "inputs/cylindrical"
PANORAMA_360_FOLDER = "inputs/360"

F_FRACTION_OF_WIDTH = 0.75
RESULTING_RECTIFICATION_WIDTH = 800
RESULTING_RECTIFICATION_HEIGHT = 600
RECTIFICATION_BORDER_FRACTION = 0.1

#================================================================================
# THE CODE ITSELF
#================================================================================

from skimage.draw import polygon
import numpy as np
import matplotlib.pyplot as plt
import math
import copy

import transforms
import pointselecter
import imageio

#================================================================================

def defineMergedImageDimensions(image_1, image_2, H):
	image_1_height = image_1.shape[0]
	image_2_height = image_2.shape[0]
	image_1_width = image_1.shape[1]
	image_2_width = image_2.shape[1]

	img1_pts = [(0,0),(image_1_width-1,0),(0,image_1_height-1),(image_1_width-1,image_1_height-1)]
	img2_pts = [(0,0),(image_2_width-1,0),(0,image_2_height-1),(image_2_width-1,image_2_height-1)]

	trans_img1_pts = []
	for pt in img1_pts:
		trans_img1_pts.append(transforms.applyHomogTransformation(H,pt))

	pts_to_check = trans_img1_pts + img2_pts
	pts_x, pts_y = zip(*pts_to_check)

	top_left = (min(pts_x),min(pts_y))
	bottom_right = (max(pts_x),max(pts_y))

	return top_left, bottom_right

def generateDistMask(W,H):
	mask = np.zeros((H,W))
	pixels = np.where(mask == 0)
	for y,x in zip(pixels[0],pixels[1]):
		mask[y,x] = min([1,min([y,H-y,x,W-x])**0.90/(0.1*max([W,H])/2)])

	return mask

def fill(source,target,T,current_source_mask=None):

	source_height = source.shape[0]
	source_width = source.shape[1]
	target_height = target.shape[0]
	target_width = target.shape[1]

	pts_source = [(0,0),(source_width-1,0),(source_width-1,source_height-1),(0,source_height-1)]

	pts_target = []
	for pt in pts_source:
		pts_target.append(transforms.applyHomogTransformation(T,pt))

	pts_target_x, pts_target_y = zip(*pts_target)

	is_someone_in = False
	for pt in pts_target:
		if pt[0] < target_width and pt[0] >= 0 and pt[1] < target_height and pt[1] >= 0:
			is_someone_in = True
			break

	if is_someone_in:
		fill_mask = np.zeros((target_height,target_width))
		rr, cc = polygon(np.array(pts_target_y), np.array(pts_target_x),fill_mask.shape)
		fill_mask[rr, cc] = 1
		target_pixels = np.where(fill_mask == 1)
	else:
		fill_mask = np.ones((target_height,target_width))
		target_pixels = np.where(fill_mask == 1)

	T_inv = np.linalg.inv(T)

	transformed_mask = np.zeros((target_height,target_width))

	if current_source_mask == None:
		blend_mask = generateDistMask(source_width,source_height)
	else:
		blend_mask = current_source_mask

	mask_height = blend_mask.shape[0]
	mask_width = blend_mask.shape[1]

	for y,x in zip(target_pixels[0],target_pixels[1]):
		original_pixel_coords = transforms.applyHomogTransformation(T_inv,(x,y))
		original_pixel_x = int(original_pixel_coords[0])
		original_pixel_y = int(original_pixel_coords[1])

		if original_pixel_x < source_width and original_pixel_x >= 0 and original_pixel_y < source_height and original_pixel_y >= 0:
			original_pixel = source[original_pixel_y,original_pixel_x]
		else:
			original_pixel = 0

		if original_pixel_x < mask_width and original_pixel_x >= 0 and original_pixel_y < mask_height and original_pixel_y >= 0:
			mask_pixel = blend_mask[original_pixel_y,original_pixel_x]
		else:
			mask_pixel = 0

		transformed_mask[y,x] = mask_pixel

		if target[y,x][0] == 0 or target[y,x][1] == 0 or target[y,x][1] == 0:
			target[y,x] = original_pixel
		else:		
			target[y,x] = original_pixel*transformed_mask[y,x] + (1-transformed_mask[y,x])*target[y,x]

	return target

def merge(image_1, image_2, H, mask = None):
	image_1_height = image_1.shape[0]
	image_2_height = image_2.shape[0]
	image_1_width = image_1.shape[1]
	image_2_width = image_2.shape[1]

	top_left_float, bottom_right_float = defineMergedImageDimensions(image_1, image_2, H)

	top_left = (int(math.floor(top_left_float[0])),int(math.floor(top_left_float[1])))
	bottom_right = (int(math.ceil(bottom_right_float[0])),int(math.ceil(bottom_right_float[1])))

	result_height = bottom_right[1] - top_left[1] + 1
	result_width = bottom_right[0] - top_left[0] + 1

	result_image = 0*np.ones((result_height,result_width,3))

	T = np.matrix([[1,0,-top_left[0]],[0,1,-top_left[1]],[0,0,1]])

	result_image = fill(image_2,result_image,T,mask)
	result_image = fill(image_1,result_image,np.dot(T,H),mask)

	empty_pixels = np.where(result_image == 0)
	result_image[empty_pixels] = 0

	return result_image

def projectOnCylinder(source,current_source_mask=None):
	source_height = source.shape[0]
	source_width = source.shape[1]

	W = source_width
	H = source_height

	F = F_FRACTION_OF_WIDTH*W

	result_image = np.zeros((H,W,3))

	target_pixels = np.where(result_image == 0)

	if current_source_mask == None:
		blend_mask = generateDistMask(source_width,source_height)
	else:
		blend_mask = current_source_mask

	transformed_mask = np.zeros((H,W))

	for y,x in zip(target_pixels[0],target_pixels[1]):
		original_pixel_coords = transforms.fromCylinderToPlane(x,y,W/2.0,H/2.0,F)
		original_pixel_x = int(original_pixel_coords[0])
		original_pixel_y = int(original_pixel_coords[1])
		if original_pixel_x < source_width and original_pixel_x >= 0 and original_pixel_y < source_height and original_pixel_y >= 0:
			original_pixel = source[original_pixel_y,original_pixel_x]
			mask_pixel = blend_mask[original_pixel_y,original_pixel_x]
		else:
			original_pixel = 0
			mask_pixel = 0
		transformed_mask[y,x] = mask_pixel
		result_image[y,x] = original_pixel

	return result_image, transformed_mask

def mergeImages(source, reference, transformFunction, mask = None):
	points_1, points_2 = pointselecter.getPoints(source,reference)

	print "Processing..... Please, wait."

	T = transformFunction(points_1,points_2)

	merged_image = merge(source,reference,T,mask)

	return merged_image

#================================================================================

def runPanoramaPlane(folder_name):

	file_names = imageio.getImageNames(folder_name)
	
	panorama = imageio.readImageFloat(file_names[0])
	for file_name in file_names[1:]:
		new_image = imageio.readImageFloat(file_name)
		panorama = mergeImages(new_image,panorama,transforms.computeHomography)

		print "Displaying the result so far:"
		figure, axis = plt.subplots(ncols=1)
		axis.imshow(panorama, vmin=0, vmax=1)
		plt.show()

	print "Displaying the final result."
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(panorama, vmin=0, vmax=1)
	plt.show()

def runPanoramaCylinder(folder_name):

	file_names = imageio.getImageNames(folder_name)
	
	panorama,mask = projectOnCylinder(imageio.readImageFloat(file_names[0]))

	for file_name in file_names[1:]:
		new_image,new_mask = projectOnCylinder(imageio.readImageFloat(file_name))

		panorama = mergeImages(new_image,panorama,transforms.computeTranslation,new_mask)

		print "Displaying the result so far:"
		figure, axis = plt.subplots(ncols=1)
		axis.imshow(panorama, vmin=0, vmax=1)
		plt.show()

	print "Displaying the final result."
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(panorama, vmin=0, vmax=1)
	plt.show()

def runPanoramaCylinderFull(folder_name):

	file_names = imageio.getImageNames(folder_name)
	
	panorama,mask = projectOnCylinder(imageio.readImageFloat(file_names[0]))
	for file_name in file_names[1:]:
		new_image,new_mask = projectOnCylinder(imageio.readImageFloat(file_name))

		panorama = mergeImages(new_image,panorama,transforms.computeTranslation,new_mask)
		
		print "Displaying the result so far:"
		figure, axis = plt.subplots(ncols=1)
		axis.imshow(panorama, vmin=0, vmax=1)
		plt.show()

	panorama_width = panorama.shape[1]

	image360 = mergeImages(panorama,panorama,transforms.computeTranslation,new_mask)

	image360_width = image360.shape[1]

	image360 = image360[:,0.25*image360_width:0.25*image360_width+(panorama_width-(2*panorama_width-image360_width))]

	print "Displaying the final result."
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(image360, vmin=0, vmax=1)
	plt.show()

def runRectification(image_name):
	img = imageio.readImageFloat(image_name)

	points = pointselecter.getFourPoints(img)

	print "Processing..... Please, wait."

	D_W = RESULTING_RECTIFICATION_WIDTH
	D_H = RESULTING_RECTIFICATION_HEIGHT
	border = RECTIFICATION_BORDER_FRACTION
	points_view = np.array([(border*D_W,border*D_H),((1-border)*D_W,border*D_H),((1-border)*D_W,(1-border)*D_H),(border*D_W,(1-border)*D_H)])

	H = transforms.computeHomography(points,points_view)

	pts_x, pts_y = zip(*points)

	result_image = 0*np.ones((D_H,D_W,3))
	fill(img,result_image,H)

	print "Displaying the final result."
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(result_image, vmin=0, vmax=1)
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

	option = raw_input("Choose an option:\n0 = Rectification\n1 = Planar Projection\n2 = Cylindrical Projection\n3 = 360 Planorama\nWrite Option Here: >> ")

	if option == "0":
		runRectification(RECTIFICATION_FILE_NAME)
	elif option == "1":
		runPanoramaPlane(PLANAR_PANORAMA_FOLDER)
	elif option == "2":
		runPanoramaCylinder(CYLINDRICAL_PANORAMA_FOLDER)
	elif option == "3":
		runPanoramaCylinderFull(PANORAMA_360_FOLDER)

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

