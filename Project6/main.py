#================================================================================
#
# University of California, Berkeley
# CS194-26 (CS294-26): Computational Photography
#
#================================================================================
#
# Project 6: "Lightfield Camera"
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
#
# Input files for this program can be found in:
# The (New) Stanford Light Field Archive (http://lightfield.stanford.edu/lfs.html)
#
#================================================================================

#================================================================================
# CONSTANTS AND SETTINGS
#================================================================================

# INPUT CONSTANTS:

# The folder path of the input files for the lightfield
INPUT_FOLDER_PATH = "../chess"

# Note:
# The program will look for files with name matching "out_yy_xx_*"
# where xx is the horizontal position of that image in the light grid
# and yy is  it the vertical position

# The number of elements of the lightfield grid
# Make sure you have at least a grid that is 2x2!
GRID_SIZE_X = 17
GRID_SIZE_Y = 17

#--------------------------------------------------------------------------------

# TWEAKING CONSTANTS:

# The size of the square patch of the image that will be used for the alignment
# Changing this has a great impact on possible cases where the proper alignment
# is not found.
# This should be large enough to cover the possible features of the images
WINDOW_SIZE = 100

# The search window size for the alignment displacement
# A displacement in the x and y will be found for each
# level of the gaussian pyramid, bounded by this constant
SEARCH_WINDOW_CTE = 20

# The number of layers of the gaussian pyramid to be calculated
# Larger numbers improve user experience and alignment, but
# consumes A LOT of memory
# Use the value -1 for calculating all levels
MAX_LAYER = 9

# Note:
# Default values for the 17x17 rectified grid of the chess board 
# from The (New) Stanford Light Field Archive (http://lightfield.stanford.edu/lfs.html)
# WINDOW_SIZE = 100
# SEARCH_WINDOW_CTE = 20
# MAX_LAYER = 9

# The starting radius (apperture size) for the application
STARTING_RADIUS = 4

#=================================================================================

#================================================================================
# THE CODE ITSELF
#================================================================================

import numpy as np
import skimage.io as skio
import skimage.transform
from skimage import img_as_float
import matplotlib.pyplot as plt
import glob
import math
import sys

#================================================================================

# Returns the image located at position (x,y) in the grid
# Float values for x and y will be truncated
def getGridImage(x,y):
	x = int(x)
	y = int(y)

	# Queries for the file out_y_x_*
	query = "%s/out_%02d_%02d_*" % (INPUT_FOLDER_PATH,y,x)
	results = glob.glob(query)
	num_results = len(results)

	if num_results == 0:
		sys.exit("Error: Image Not Found")
	elif num_results > 1:
		sys.exit("Error: Multiple Images Found")
	else:
		img = skio.imread(results[0])
		return img_as_float(img)

#--------------------------------------------------------------------------------

# Reads and puts in memory the gaussian pyramid of the input lightfield
def getGridPyramid():
	grid = []
	for y in xrange(GRID_SIZE_Y):
		print "Reading Input and Preprocessing Images: %3d%% complete..." % int(100*(y+1)/float(GRID_SIZE_Y))
		new_row = []
		for x in xrange(GRID_SIZE_X):
			img = getGridImage(x,y)
			pyramid = gaussianPyramid(img)
			new_row.append(pyramid)
		grid.append(new_row)
	return np.array(grid)

#--------------------------------------------------------------------------------

# Returns a square patch (an image) cropped from img,
# centered in p with dimensions w by w
def cropPatch(img,p,w):
	x = p[0]
	y = p[1]

	# Cases near the border or with too low resolution for w by w
	k = max(1,int(w/2))
	min_dim = min(img.shape[0],img.shape[1])
	k = min(k,min_dim/2)
	min_x = max(x-k,0)
	max_x = min(x+k,img.shape[1])
	min_y = max(y-k,0)
	max_y = min(y+k,img.shape[0])

	return img[min_y:max_y,min_x:max_x]

#--------------------------------------------------------------------------------

# Return a list that contains the layers of the gaussian pyramid of image
def gaussianPyramid(image):
	return list(skimage.transform.pyramid_gaussian(image,max_layer=MAX_LAYER))

#--------------------------------------------------------------------------------

# Translates img by dx in x and dy in y
def imgTranslate(img,dx,dy):
	result = np.roll(img, dx, axis = 1)
	result = np.roll(result, dy, axis = 0)
	return result

#--------------------------------------------------------------------------------

# Compute the SSD between img_1 and img_2
def ssd(img_1, img_2):
	return np.sum(((img_1 - img_2)**2))

#--------------------------------------------------------------------------------

# Returns the best alignment (a tuple of values, one for x and one for y) that
# correspond to the values you need to translate the target to get the least
# SSD between the images
def bestAlignment(ref, target):
	best_value = float("inf")
	best_dx = 0
	best_dy = 0

	width_search_window = min(SEARCH_WINDOW_CTE,ref.shape[0]/2)
	height_search_window = min(SEARCH_WINDOW_CTE,ref.shape[1]/2)

	# Searches in the displacement search window
	for dx in reversed(xrange(-width_search_window,width_search_window)):
		for dy in reversed(xrange(-height_search_window,height_search_window)):
			traslated_target = imgTranslate(target,dx,dy)
			distance = ssd(ref,traslated_target)
			# Get the smallest distances
			if distance < best_value:
				best_value = distance
				best_dx = dx
				best_dy = dy

	return (best_dx, best_dy)

#--------------------------------------------------------------------------------

# Returns the best alignment for the first layer (original images) of the
# gaussian pyramids of the reference and target by using the rest of the pyramid
# for speeding up things
# The alignment is made according to a point. The distance that we want to
# minimize is the SSD of a square patch around the given point
def alignByPyramids(py_ref,py_target,point):
	len_py_ref = len(py_ref)
	len_py_target = len(py_target)
	if len_py_ref != len_py_target:
		sys.exit("Error: Pyramids do not have the same size.")

	best_dx, best_dy = 0, 0
	for ref,target,level_number in reversed(zip(py_ref,py_target,range(len_py_ref))):
		best_dx *= 2
		best_dy *= 2

		ptx = point[0]/(2**level_number) + best_dx
		pty = point[1]/(2**level_number) + best_dy

		adjusted_target = imgTranslate(target,best_dx,best_dy)

		w = WINDOW_SIZE
		ref_patch = cropPatch(ref,(ptx,pty),w)
		target_patch = cropPatch(adjusted_target,(ptx,pty),w)

		new_dx, new_dy = bestAlignment(ref_patch,target_patch)

		best_dx += new_dx
		best_dy += new_dy

	return (best_dx,best_dy)

#--------------------------------------------------------------------------------

# The manhattam distance between two points p and q
def manDist(p,q):
	return abs(p[0] - q[0]) + abs(p[1] - q[1])

#--------------------------------------------------------------------------------

# Given the pyramidal grid of the lightfield and the spacing dx and dy between the grid
# images, averages all the images of the given level in the pyramids that are within the given radius
# of the given center
def avgAlignedGridByLevel(grid,dx,dy,center,radius,level):
	average = np.zeros(grid[0,0][level].shape)

	dx = dx/(2**level)
	dy = dy/(2**level)

	images_used = GRID_SIZE_Y*GRID_SIZE_X

	for y in xrange(GRID_SIZE_Y):
		for x in xrange(GRID_SIZE_X):
			# Discard images out of range
			if manDist((x,y),center) > radius:
				images_used -= 1
				continue

			# Get image, align, average
			img = grid[y,x][level]
			trans_x = int(-x*dx)/(2**level)
			trans_y = int(-y*dy)/(2**level)
			img_aligned = imgTranslate(img,trans_x,trans_y)
			average += img_aligned
	average /= images_used
	return average


#================================================================================
# THE APPLICATION
#================================================================================

# Parameters to be maintened during out application
radius = STARTING_RADIUS
dx = 0.0
dy = 0.0
axis = None
py_ref = None
py_d_ref = None
grid = None
center = None
image = None

#--------------------------------------------------------------------------------

# Function to be called whenever the user clicks or scrolls on the image
def clickOnImage(event):

	# Global Stuff
	global dx
	global dy
	global radius
	global axis
	global py_ref
	global py_d_ref
	global grid
	global center
	global image

	update = False

	# Left Click
	if event.button == 1:
		if event.inaxes == axis:
			# Update grid displacement according to new feature
			point = (int(event.xdata),int(event.ydata))
			delta_x,delta_y = alignByPyramids(py_ref,py_d_ref,point)

			dx = float(-delta_x)/float(GRID_SIZE_X)
			dy = float(-delta_y)/float(GRID_SIZE_Y)

			update = True

	# Scroll Up
	elif event.button == 'up':
		if event.inaxes == axis:
			# Decrease radius
			radius += event.step
			if radius < 1:
				radius = 1
			update = True
	
	# Scroll Down
	elif event.button == 'down':
		if event.inaxes == axis:
			# Increase radius
			radius += event.step
			if radius < 1:
				radius = 1
			update = True

	# If something changed, update plot
	if update:
		print "=================="
		print "Showing image for setting:"
		print "Grid dx = " + str(dx)
		print "Grid dy = " + str(dy)
		print "Aperture Radius = " + str(radius)
		print "=================="
		print "Click to Focus, Scroll to Change Aperture Size"
		print "=================="

		# Show low resolution versions of the output first, to improve the user experice
		for level in reversed(xrange(len(grid[0,0]))):
			avg = avgAlignedGridByLevel(grid,dx,dy,center,radius,level)
			image.set_data(avg)
			plt.draw()

#--------------------------------------------------------------------------------

# The application code/loop itself
def run():

	# Global Stuff
	global dx
	global dy
	global radius
	global axis
	global py_ref
	global py_d_ref
	global grid
	global center
	global image

	# Read inputs
	grid = getGridPyramid()

	# Define the reference images and the center images
	py_ref = grid[0,0]
	radius = STARTING_RADIUS
	center = (GRID_SIZE_X/2,GRID_SIZE_Y/2)
	py_d_ref = grid[(GRID_SIZE_Y-1),(GRID_SIZE_X-1)]

	# Get a point to start
	point = (0,0)
	delta_x,delta_y = alignByPyramids(py_ref,py_d_ref,point)

	dx = float(-delta_x)/float(GRID_SIZE_X)
	dy = float(-delta_y)/float(GRID_SIZE_Y)

	avg = avgAlignedGridByLevel(grid,dx,dy,center,radius,0)

	# Create the plot for the application
	figure, axis = plt.subplots(ncols=1)
	image = axis.imshow(avg, vmin=0, vmax=1)

	# Sets the mouse events
	listener_button_press = figure.canvas.mpl_connect('button_press_event', clickOnImage)
	listener_scroll = figure.canvas.mpl_connect('scroll_event', clickOnImage)

	print "=================="
	print "Click to Focus, Scroll to Change Aperture Size"
	print "=================="
	plt.show()
	plt.draw()

	# Close everything
	figure.canvas.mpl_disconnect(listener_button_press)
	figure.canvas.mpl_disconnect(listener_scroll)
	plt.close()


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
	print ">> Project 6: Lightfield Camera"
	print ">> Student: Ian Albuquerque Raymundo da Silva"
	print ">> Email: ian.albuquerque@berkeley.edu"
	print "==============================================="
	print "==============================================="
	print "============== GO BEARS!!!!!!!! ==============="
	print "==============================================="
	print "==============================================="

	run()

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

