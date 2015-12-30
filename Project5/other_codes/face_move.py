import math
import sys
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from skimage import img_as_float
from skimage.draw import polygon

#faces = ["angry","blob","bored","cheek_left","cheek_right","crazy","creepy","duck","happy","intrigued","normal","sad","surprise","weird","why"]
faces = ["angry","blink_left","blink_right","cheek_left","cheek_right","fish","happy","idk","weird","what","wow"]

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
				marker=plt.Circle((event.xdata,event.ydata),0.005*image_1_width,color='r')
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

	points = {}
	input_images = {}
	for face in faces:
		input_image = skio.imread("faces/"+face+".JPG")
		input_images[face] = input_image
		points[face] = getPoints(input_image)

	target_image = skio.imread("orc.jpg")
	target_points = getPoints(target_image)

	#corresp = raw_input("Insert name of corresponding image:")
	corresp = "angry"

	delanayTriangulation = Delaunay(target_points)
	target_triangles = delanayTriangulation.simplices

	for desired in faces:
		#desired = raw_input("Insert name of desired image:")

		dif_points = np.array(points[desired]) - np.array(points[corresp])
		end_points = np.array(target_points)+dif_points

		alpha = 0
		N = 15
		d = 1.0/float(N)
		for i in range(N+1):
			alpha = i*d
			points_inter = (1.0-alpha)*np.array(target_points) + alpha*end_points
			output_image = np.array(morph(target_image,target_points,points_inter,target_triangles))
			skio.imsave("out/"+desired+str(i)+".jpg", output_image)

		output_image = np.array(morph(target_image,target_points,np.array(target_points)+dif_points,target_triangles))

		figure, axes = plt.subplots(ncols=2)
		axes[0].imshow(target_image, vmin=0, vmax=1)
		axes[1].imshow(output_image, vmin=0, vmax=1)
		plt.show()
		plt.draw()
		plt.close()

