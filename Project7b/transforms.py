import numpy as np
import sys
import math

def fromCylinderToPlane(x_cyl,y_cyl,x_c,y_c,f):
	if f==0:
		sys.exit("Error: f equals zero.")

	theta = (x_cyl - x_c)/f
	h = (y_cyl - y_c)/f

	x_hat = math.sin(theta)
	y_hat = h
	z_hat = math.cos(theta)

	x = f*x_hat/z_hat + x_c
	y = f*y_hat/z_hat + y_c

	return (x,y)

def computeHomography(pts_source, pts_transformed):
	num_pts_source = len(pts_source)
	num_pts_transformed = len(pts_transformed)

	if num_pts_source != num_pts_transformed:
		sys.exit("Error: Number of points given to homography computation do not match.")
	if num_pts_source < 4 or num_pts_transformed < 4:
		sys.exit("Error: Not enough points were given to calculate homography.")

	A = []
	b = []
	for pt_src, pt_trans in zip(pts_source,pts_transformed):
		x = pt_src[0]
		y = pt_src[1]
		xp = pt_trans[0]
		yp = pt_trans[1]

		A.append([x,y,1,0,0,0,(-x*xp),(-y*xp)])
		A.append([0,0,0,x,y,1,(-x*yp),(-y*yp)])
		b.append(xp)
		b.append(yp)

	np_A = np.array(A)
	np_b = np.array(b)

	x = np.linalg.lstsq(np_A,np_b)[0]

	H = [[x[0],x[1],x[2]],[x[3],x[4],x[5]],[x[6],x[7],1]]

	return np.matrix(H)


def computeTranslation(pts_source, pts_transformed):
	num_pts_source = len(pts_source)
	num_pts_transformed = len(pts_transformed)

	if num_pts_source != num_pts_transformed:
		sys.exit("Error: Number of points given to homography computation do not match.")
	if num_pts_source < 1 or num_pts_transformed < 1:
		sys.exit("Error: Not enough points were given to calculate translation.")

	A = []
	b = []
	for pt_src, pt_trans in zip(pts_source,pts_transformed):
		x = pt_src[0]
		y = pt_src[1]
		xp = pt_trans[0]
		yp = pt_trans[1]

		A.append([1,0])
		A.append([0,1])
		b.append(xp-x)
		b.append(yp-y)

	np_A = np.array(A)
	np_b = np.array(b)

	x = np.linalg.lstsq(np_A,np_b)[0]

	T = [[1,0,x[0]],[0,1,x[1]],[0,0,1]]

	return np.matrix(T)

def toHomog(pt):
	return (pt[0],pt[1],1)

def fromHomog(homog_pt):
	if homog_pt[2] == 0:
		sys.exit("Error: Point mapped to infinity when homogeneous.")
	return (homog_pt[0]/homog_pt[2],homog_pt[1]/homog_pt[2])

def applyHomogTransformation(T,pt):
	homog_pt = toHomog(pt)
	trans_pt_homog = np.array(np.dot(T,homog_pt))
	return fromHomog(trans_pt_homog[0])