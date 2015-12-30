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
# seam_carving.py
#
# This file contains all the necessary functions for the preprocessing and the
# online parts of the assignment
#================================================================================


from skimage import filter
from skimage import img_as_float
import numpy as np
import skimage.io as skio
import time

#----------------------------------------------------------------------

# Displays an image
def display(image):
	skio.imshow(image)
	skio.show()

# Retrieve the color channels of an image as a list
def getColorChannels(image):
	return [image[:,:,0],image[:,:,1],image[:,:,2]]

# Returns the transpose of an image
def transposeImage(image):
	color_channels = getColorChannels(image)

	new_channels = []
	for channel in color_channels:
		new_channels.append(channel.transpose())

	transp_image = np.dstack(new_channels)
	return transp_image

# Converts an colored image to a single scaled gray image
def colorToGray(image):
	color_channels = getColorChannels(image)
	return np.clip(color_channels[0]*0.299 + color_channels[1]*0.587 + color_channels[2]*0.114, 0.0, 1.0)

#----------------------------------------------------------------------

# Linearly adjust the image's values to fit in the [0,1] range
def normalize(image):
	positive_image = image - np.min(image)
	return positive_image / np.max(positive_image)

# Returns an absolute version of the values of the image in the range of [0,1]
def absolutize(image):
	image = np.abs(image)
	return normalize(image)

# Calculates the energy function for one channel
def energyFunctionChannel(channel):
	# The energy function of a single-scaled image (channel)
	# is the absolute and normalized value of the
	# image after an laplacian filter
	# (The energy function will be higher on edges)
	blurred_channel = filter.gaussian_filter(channel, 2)
	high_freq = channel - blurred_channel
	absol = absolutize(high_freq)
	return absol

# Calculates the energy function of an image considering an energy mask
def energyFunction(color_channels, energy_mask_channels):
	# The energy function of an image is the sum of the energy function
	# of its channels

	# The energy mask will be considered as weights for each pixel
	return energyFunctionChannel(color_channels[0])*energy_mask_channels[0] + energyFunctionChannel(color_channels[1])*energy_mask_channels[0] + energyFunctionChannel(color_channels[2])*energy_mask_channels[2]

# Calculates the energy function of an image
def energyFunctionNoMask(color_channels):
	# The energy function of an image is the sum of the energy function
	# of its channels
	return energyFunctionChannel(color_channels[0]) + energyFunctionChannel(color_channels[1]) + energyFunctionChannel(color_channels[2])

#----------------------------------------------------------------------

# (Online) Reduces the given image by the specified amount "reduce_by" 
# given the pixels orders to be removed passed as "order"
def reduceImageHorizontally(image, order, reduce_by):

	# Retrieve the color channels of the image
	color_channels = getColorChannels(image)

	# This will be the result of the operation
	new_color_channels = []

	# If the reduce amount is positive, the output will be a smaller image
	# (Seam Carving)
	if reduce_by >=0:
		# For each channel
		for channel in color_channels:
			new_channel = []
			# Iterate through all lines of the channel
			for y in range(channel.shape[0]):

				# Finds the indexes of all pixels that need to be removed
				indexes_to_remove = []

				# For each pixel in the line
				for x in range(order.shape[1]):
					# If the order of that pixel results in its removal of the image
					if order[y,x] < reduce_by:
						# Add that to the list of pixels to be removed
						indexes_to_remove.append(x)

				# Remove all pixels found in the previous step
				new_channel.append(np.delete(channel[y,:],indexes_to_remove))

			# Appends the new color channel created
			new_color_channels.append(np.array(new_channel))


	# If the reduce amount is negative, the output will be a larger image
	# (Seam Insertion)
	else:
		# Lets consider the absolute value of "reduce_by"
		reduce_by = reduce_by * -1

		# For each color channel
		for channel in color_channels:
			new_channel = []

			# Iterate through each line
			for y in range(channel.shape[0]):

				# The new row will have its base as a copy of the line
				new_row = channel[y,:].copy()

				# Iterate the row backwards, looking for places to insert pixels
				for x in reversed(range(order.shape[1])):

					# If you find a place to insert a pixel
					if order[y,x] < reduce_by:

						# The pixel to be inserted will be the average of the horizontal pixels around it
						pixel_to_insert = sum([channel[y,min([order.shape[1]-1,x+1])], channel[y,max([0,x-1])], channel[y,x]])/3.0

						# Insert the pixel in its place
						new_row = np.insert(new_row, x, pixel_to_insert)

				# Append the new row into the new channel
				new_channel.append(new_row)

			# Append the new channel
			new_color_channels.append(np.array(new_channel))

	# Stack all channels into a single image
	output_image = np.dstack(new_color_channels)

	return output_image

#----------------------------------------------------------------------

# (Precomputing) Given an energy function, computes the minimum cost paths
# through each pixel using Dynamid Programming
def calculatePathDP(energy):
	dp = np.zeros(energy.shape)

	# The base case for the dynamic programming is the top line of the image that will
	# have its values corresponding to its energy function
	dp[0,:] = np.copy(energy[0,:])

	# For each pixel in the dynamic programming matrix
	it = np.nditer(dp, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		y,x = it.multi_index
		# The value of the path will be the energy function of that pixel added to the
		# minimum cost of the pixels above (diagonal left top, vertical top and diagonal right top)
		it[0] = energy[y,x] + min([dp[y-1,max(0,x-1)],dp[y-1,x],dp[y-1,min(x+1,energy.shape[1]-1)]])
		it.iternext()

	return dp

#----------------------------------------------------------------------

# (Precomputing) Backtracks the path found through the Dynamid Programming
# Returns an array containt the indexes of the collum to be deleted for each row
def backtrack(dp):
	smallest_index = -1

	indexes_to_remove = []

	# Iterates throgh rows of the matrix from the bottom to the top
	for row in reversed(dp):
		if smallest_index == -1:
			# Starts by taking the lowest pixel value
			smallest_index = np.argmin(row)
		else:
			# On subsequent rows, take the minimum pixel values of immediately above the
			# last picked pixel
			begin = max(0,smallest_index-1)
			end = min(smallest_index+1,dp.shape[1]-1)+1
			smallest_index = np.argmin(row[begin:end]) + begin

		# Append that index in that row to be removed
		indexes_to_remove.append(smallest_index)

	# The actual path from the top to bottom will be reversed. Fix that.
	indexes_to_remove.reverse()

	return indexes_to_remove

#----------------------------------------------------------------------

# (Precomputing) Shrinks an image and its energy mask horizontally by one pixel,
# given an energy function of the image
def shrinkByOne(color_channels, energy, energy_mask_channels):

	# Calculates the lowest energy path with DP
	dp = calculatePathDP(energy)

	# Backtracks the path
	indexes_to_remove = backtrack(dp)

	# Removes the path from the image and energy mask

	new_color_channels = []
	new_energy_mask_channels = []

	# Iterates through each channel
	for channel, energy_channel in zip(color_channels,energy_mask_channels):
		new_channel = []
		new_energy_channel = []

		# For each row
		for y in range(channel.shape[0]):

			# Delete the corresponding index found
			new_channel.append(np.delete(channel[y,:],indexes_to_remove[y]))
			new_energy_channel.append(np.delete(energy_channel[y,:],indexes_to_remove[y]))

		# Updates the new channel and new energy mask
		new_energy_channel = np.array(new_energy_channel)
		new_channel = np.array(new_channel)
		new_color_channels.append(new_channel)
		new_energy_mask_channels.append(new_energy_channel)

	return new_color_channels, indexes_to_remove, new_energy_mask_channels

# (Precomputing) Shrinks an image horizontally by one pixel given an energy function
def shrinkByOneNoMask(color_channels, energy):

	# Calculates the lowest energy path with DP
	dp = calculatePathDP(energy)

	# Backtracks the path
	indexes_to_remove = backtrack(dp)

	new_color_channels = []

	# Iterates through each channel
	for channel in color_channels:
		new_channel = []

		# For each row
		for y in range(channel.shape[0]):

			# Delete the corresponding index found
			new_channel.append(np.delete(channel[y,:],indexes_to_remove[y]))

		# Updates the new channel
		new_channel = np.array(new_channel)
		new_color_channels.append(new_channel)

	return new_color_channels, indexes_to_remove

#----------------------------------------------------------------------

# (Precomputing) Given an image and the pixels removed in each step of
# the precomputing, retrieve the order of relevance of each pixel
# in the image
def retrieveOrder(image, marks):

	# Creates an empty order array
	order = np.zeros((image.shape[0],1))

	r = 0
	# For each array of indexes removed (backtracking)
	for mark in marks:
		if r > 0:
			new_order = []
			# Itereate through the array
			for i in range(len(mark)):
				# Insert into the updated order matrix the deletions that were made
				new_order.append(np.insert(order[i],mark[i],r))
			new_order = np.array(new_order)
			# Updates order array to continue to next step
			order = new_order
		r = r + 1

	# The first removals will have a high value. This fixes that, so the first
	# pixel order value is zero and the last is the highest
	order = len(marks) - order
	return order

#----------------------------------------------------------------------

# (Precomputing) Saves the order of relevance matrix as an ".ord" file
def saveOrderFile(output_file, order):

	# Outputs the shape of the order matrix
	output_file.write(str(order.shape[0]) + "\n")
	output_file.write(str(order.shape[1]) + "\n")

	# Outputs each cell of the order matrix
	for cell in np.nditer(order, order='C'):
		output_file.write(str(cell) + "\n")

#----------------------------------------------------------------------

# (Precomputing) Computes the order of relevance of each pixel in an image
# given an energy mask, considering that the reduction in size will
# be performed horizontally
def computeHorizontalOrder(image, energy_mask):

	# Makes sure that we will not change the input data
	image = image.copy()
	energy_mask = energy_mask.copy()

	# Retrieves the color channels of the image and energy mask
	color_channels = getColorChannels(image)
	energy_mask_channels = getColorChannels(energy_mask)

	# For estimating processing time
	t = time.time()
	avg_step = 0
	marks = []

	# For each possible width of the image
	for w in range(image.shape[1]):

		# For estimating processing time
		step_t = time.time()

		# Compute the energy function of the image with the energy mask)
		energy = energyFunction(color_channels, energy_mask_channels)

		# Updates the image and energy_mask by shrinking horizontally by one pixel
		color_channels, indexes_to_mark, energy_mask_channels = shrinkByOne(color_channels,energy,energy_mask_channels)

		# Saves the delete information for recovering pixel order later
		marks.append(indexes_to_mark)

		# For estimating processing time
		current_completion = 100.0 * float(w) / float(image.shape[1])
		this_step = time.time() - step_t
		avg_step = float(avg_step * (w) + this_step) / float(w+1)
		print "Progress: %6.02f %%. (About %7.02f seconds remaining)" % (current_completion,  (image.shape[1] - w)*avg_step)

	print "Done! Retrieving pixel oder..."

	# Reverses the pixel removal order, so the last deletions are first
	marks.reverse()

	# Retrieves the deletion order
	order = retrieveOrder(image, marks)

	print "Computing completed in %7.02f seconds." % (time.time() - t)

	return order

# (Precomputing) Computes the order of relevance of each pixel in an image
# considering that the reduction in size will be performed horizontally
def computeHorizontalOrderNoMask(image):

	# Makes sure that we will not change the input data
	image = image.copy()

	# Retrieves the color channels of the image
	color_channels = getColorChannels(image)

	# For estimating processing time
	t = time.time()
	avg_step = 0
	marks = []

	for w in range(image.shape[1]):
		# For estimating processing time
		step_t = time.time()

		# Compute the energy function of the image
		energy = energyFunctionNoMask(color_channels)

		# Updates the image by shrinking horizontally by one pixel
		color_channels, indexes_to_mark = shrinkByOneNoMask(color_channels,energy)
		
		# Saves the delete information for recovering pixel order later
		marks.append(indexes_to_mark)

		# For estimating processing time
		current_completion = 100.0 * float(w) / float(image.shape[1])
		this_step = time.time() - step_t
		avg_step = float(avg_step * (w) + this_step) / float(w+1)
		print "Progress: %6.02f %%. (About %7.02f seconds remaining)" % (current_completion,  (image.shape[1] - w)*avg_step)

	print "Done! Retrieving pixel oder..."

	# Reverses the pixel removal order, so the last deletions are first
	marks.reverse()

	# Retrieves the deletion order
	order = retrieveOrder(image, marks)

	print "Computing completed in %7.02f seconds." % (time.time() - t)

	return order
