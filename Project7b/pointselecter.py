import numpy as np
import matplotlib.pyplot as plt
import sys

def getFourPoints(image):

	print "Please, select four points on the image."

	# Get the dimensions
	image_height = image.shape[0]
	image_width = image.shape[1]

	points = []

	# Show the images
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(image, vmin=0, vmax=1)

	# Event for clicking on the images
	def clickOnImage(event):
		if event.button == 1:
			# Image one selected
			if event.inaxes == axis:
				points.append((int(event.xdata),int(event.ydata)))
				marker=plt.Circle((event.xdata,event.ydata),0.006*image_width,color='r')
				axis.add_artist(marker)

		plt.show(block=False)
		plt.draw()

		if len(points) == 4:
			plt.close()

	# Set up the events
	listener_button_press = figure.canvas.mpl_connect('button_press_event', clickOnImage)

	# Show imaves
	plt.show()
	plt.draw()

	# Close everything
	figure.canvas.mpl_disconnect(listener_button_press)

	plt.close()

	# Check if points are OK
	if len(points) != 4:
		sys.exit("Error: The number of points selected is not 4.")

	return np.array(points)

def getPoints(image_1, image_2):

	print "Please, click on the images to choose as many correspondence points as you can."

	# Get the dimensions
	image_1_height = image_1.shape[0]
	image_2_height = image_2.shape[0]
	image_1_width = image_1.shape[1]
	image_2_width = image_2.shape[1]

	# Used for controlling the alternate input between images
	click_on_image_1 = [True]

	points_1 = []
	points_2 = []

	# Show the images
	figure, axes = plt.subplots(ncols=2)
	axes[0].imshow(image_1, vmin=0, vmax=1)
	axes[1].imshow(image_2, vmin=0, vmax=1)

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
		sys.exit("Error: The number of points selected in both images is not the same!")

	return np.array(points_1), np.array(points_2)