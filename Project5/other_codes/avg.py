import math
import sys
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skimage import img_as_float
from skimage.draw import polygon

def getPoints(image):

	image_1_height = image.shape[0]
	image_1_width = image.shape[1]

	figure, axes = plt.subplots(ncols=1)
	axes.imshow(image, vmin=0, vmax=1)

	points_1 = [(0,0),(image_1_width-1,0),(0,image_1_height-1),(image_1_width-1,image_1_height-1)]

	lines = [[],[]]

	def onKeyPress(event):
		if event.key == ' ':
			plt.close()

	def clickOnImage(event):
		if event.button == 1:
			if event.inaxes == axes:
				points_1.append((int(event.xdata),int(event.ydata)))
				marker=plt.Circle((event.xdata,event.ydata),0.01*image_1_width,color='r')
				axes.add_artist(marker)

				plt.show(block=False)
				plt.draw()

	listener_button_press = figure.canvas.mpl_connect('button_press_event', clickOnImage)
	listener_key_press = figure.canvas.mpl_connect('key_press_event', onKeyPress)

	plt.show()
	plt.draw()

	figure.canvas.mpl_disconnect(listener_button_press)
	figure.canvas.mpl_disconnect(listener_key_press)

	plt.close()

	return np.array(points_1)
	
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

def morph(origin_image,origin_points,destination_points,triangles):
	
	output_image = origin_image*0

	prog = 0
	maxi = len(triangles) 
	for triangle in triangles:
		prog = prog + 1
		print str(prog*100.0/maxi) + " percent completed"
		T = computeAffine(triangle,origin_points,destination_points)
		img = np.zeros((origin_image.shape[0], origin_image.shape[1]), dtype=np.uint8)
		x = np.array([destination_points[triangle[0]][0], destination_points[triangle[1]][0], destination_points[triangle[2]][0]])
		y = np.array([destination_points[triangle[0]][1], destination_points[triangle[1]][1], destination_points[triangle[2]][1]])
		rr, cc = polygon(y, x)
		img[rr, cc] = 1
		triangle_pixels = np.where(img == 1)
		for y,x in zip(triangle_pixels[0],triangle_pixels[1]):
			hom = np.array([x,y,1.0])
			origin_point_hom = np.array(np.dot(T,hom))[0]
			output_image[y,x] = origin_image[int(origin_point_hom[1]),int(origin_point_hom[0])]

	return output_image

if __name__ == '__main__':

	points = []
	input_images = []
	n_p = 39
	for img in range(1,20):
		input_image = skio.imread("faces/"+str(img)+"-13.jpg")
		input_images.append(input_image)
		do = "f"
		while do != "t":
			p = getPoints(input_image)
			do = raw_input("good?")
			if len(p) != 39+4:
				do = "f"
		points.append(p)

	my_face = skio.imread("my_face.jpg")
	my_points = np.array(getPoints(my_face))

	points_inter = points[0] - points[0]
	for p in points:
		points_inter = points_inter + p
	points_inter = points_inter / len(points)

	delanayTriangulation = Delaunay(points_inter)
	triangles = delanayTriangulation.simplices

	morphed_images = []
	for image, pt in zip(input_images,points):
		image_morphed = img_as_float(np.array(morph(image,pt,points_inter,triangles)))
		morphed_images.append(image_morphed)

	output_image = input_images[0]*0.0
	for img in morphed_images:
		output_image = output_image + img
	output_image = output_image / len(morphed_images)

	skio.imsave("avg.jpg", output_image)

	me_in_avg = np.array(morph(my_face,my_points,points_inter,triangles))
	skio.imsave("me_in_avg.jpg", me_in_avg)

	avg_in_me = np.array(morph(output_image,points_inter,my_points,triangles))
	skio.imsave("avg_in_me.jpg", avg_in_me)

	a = 0.0
	i = 0
	while a <= 2.0:
		extrapolated_points = my_points + a*(my_points - points_inter)
		me_caricature = np.array(morph(my_face,my_points,extrapolated_points,triangles))
		skio.imsave("out/car_"+str(i)+".jpg", me_caricature)
		i = i + 1
		a = a + 0.1


