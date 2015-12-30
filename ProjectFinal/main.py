#================================================================================
#
# University of California, Berkeley
# CS194-26 (CS294-26): Computational Photography"
#
#================================================================================
#
# Final Project: "Video Matching"
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

import pylab
import imageio
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage import img_as_float
from skimage.transform import rescale
import random
import math
import harris
from skimage import filter

INIT_TAR_FRAME = 0
END_TAR_FRAME = 600
INIT_REF_FRAME = 100
END_REF_FRAME = 300

def ssd(img_1,img_2):
	if img_1.shape != img_2.shape:
		raise TypeError("Wrong Shapes!")
	return np.sum((img_1-img_2)**2)

def bw(img):
	return 0.21*img[:,:,0] + 0.72*img[:,:,1] + 0.07*img[:,:,2]

def trip(img):
	return np.dstack([img,img,img])

def getLowResolution(video_reader,start_frame,end_frame,scale_factor):
	result_video = []
	original_indexes = []
	for index in xrange(start_frame,end_frame):
		print index
		img = video_reader.get_data(index)
		result_video.append(rescale(bw(img_as_float(img)),scale_factor))
		original_indexes.append(index)
	return result_video, original_indexes

def saveVideo(file_name,video_reader,desired_fps,indexes):
	writer = imageio.get_writer(file_name, fps=desired_fps)
	for index in indexes:
		print index
		writer.append_data(video_reader.get_data(index))
	writer.close()

def saveFrames(file_name,frames,desired_fps):
	writer = imageio.get_writer(file_name, fps=desired_fps)
	for index in xrange(len(frames)):
		print index
		writer.append_data(frames[index])
	writer.close()

def lot(numpy_probability_array):
	if numpy_probability_array.size == 1:
		return 0

	random_number = random.random()
	sum_until_now = 0.0
	for index, probability in enumerate(numpy_probability_array):
		sum_until_now += probability
		if sum_until_now > random_number:
			return index

	return -1

def w(k):
	return 1/(math.fabs(k)+1)
	#return 1

def getProbabilities(frames, indexes, video_reader, fps):
	num_frames = len(frames)
	distance = np.zeros((num_frames,num_frames))
	for i1, frame1 in enumerate(frames):
		for i2, frame2 in enumerate(frames):
			print (i1,i2)
			dist = ssd(frame1,frame2)
			distance[i1,i2] = dist

	N = int(raw_input("Type N (2) = "))

	# -------------
	# Calculating costs out of distances

	num_frames_rectified = num_frames - 1 - 2*N
	cost = np.zeros((num_frames_rectified,num_frames_rectified))
	for i1 in xrange(num_frames_rectified):
		for i2 in xrange(num_frames_rectified):
			cost[i1,i2] = 0.0
			sum_weights = 0.0
			for k in xrange(-N,N):
				cost[i1,i2] += w(k)*distance[i1+k+1,i2+k]
				sum_weights += w(k)
			cost[i1,i2] /= sum_weights

	cost /= np.max(cost)

	while(True):
		sigma = float(raw_input("Type sigma = "))
		LMR = int(raw_input("Type LMR = "))
		alpha = float(raw_input("Type alpha (0.99) = "))
		tolerance = float(raw_input("Type tolerance (0.1) = "))
		p = float(raw_input("Type p (1.1) = "))

		# -------------
		# Q Learning

		update_dif = float("inf")

		new_cost = np.copy(cost)
		while update_dif > tolerance:
			update_dif = 0

			m = []
			for j in xrange(num_frames_rectified):
				m.append(min([new_cost[j,k] for k in xrange(num_frames_rectified)]))

			before_values = np.copy(new_cost)

			for j in xrange(num_frames_rectified):
				new_cost[:,j] = cost[:,j]**p + alpha*m[j]

			update_dif = np.sum(np.absolute(new_cost - before_values))

			#print (update_dif,np.sum(new_cost))
			print update_dif

			stop_q_learning = False
			new_probability = np.exp(-new_cost/sigma)
			for index in xrange(num_frames_rectified):
				if np.sum(new_probability[index,:]) == 0:
					stop_q_learning = True

			if stop_q_learning:
				break
			else:
				probability = new_probability

		figure, axis = plt.subplots(ncols=1)
		axis.imshow(trip(new_cost/np.max(new_cost)), vmin=0, vmax=1, interpolation="nearest")
		plt.show(block=True)

		# -------------
		# Locam Maxima Finding

		if LMR != 0:
			prob_local = np.zeros((num_frames_rectified,num_frames_rectified))
			for i1 in xrange(num_frames_rectified):
				for i2 in xrange(num_frames_rectified):
					lim_s = max([0,i2-LMR])
					lim_b = min([num_frames_rectified-1,i2+LMR])
					if probability[i1,i2] == max(probability[i1,lim_s:lim_b]):
						prob_local[i1,i2] = sum(probability[i1,lim_s:lim_b])
			probability = prob_local

		# -------------
		# Prob Normalization

		for index in xrange(num_frames_rectified):
			probability[index,:] = probability[index,:]/np.sum(probability[index,:])

		figure, axis = plt.subplots(ncols=1)
		axis.imshow(trip(probability/np.max(probability)), vmin=0, vmax=1, interpolation="nearest")
		plt.show(block=True)

		# -------------

		min_num_frames = int(raw_input("Frames to Generate:"))

		fr_index = 0
		result_indexes = [indexes[fr_index+N]]
		count = 0
		while(count < min_num_frames):
			count += 1
			next_frame = lot(probability[fr_index,:])
			result_indexes.append(indexes[next_frame+N])
			fr_index = next_frame

		print result_indexes

		# -------------

		saveVideo("out.mp4",video_reader,fps,result_indexes)

		answer = raw_input("Return function?")
		if answer == 'y':
			return probability, N


HARRIS_TRESHOLD_RELATIVE = 0.1
DESCRIPTOR_SHAPE_XY = (16,16)
NUM_POINTS_TO_FIND = 1000
MIN_RADIUS_POINTS = 6
RUSSIAN_GRANDMA_TRESHOLD = 0.5
EDGE_DISCARD = 8

def computeCornerResponse(img, radius, max_number_points,harris_threshold_rel):
	max_dimension = max([img.shape[0],img.shape[1]])

	gray_img = img

	edge_discard = EDGE_DISCARD
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
				# print "Found " + str(len(indexes_selected)) + " out of " + str(max_number_points) + " points expected."
				if len(indexes_selected) >= max_number_points:
					break
		if len(indexes_selected) >= max_number_points:
			break

	points = []
	for index in indexes_selected:
		points.append((interest_points[1][index],interest_points[0][index]))

	return points

def extractDescriptors(img,points,shape):
	gray_img = img

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
	dist_sum = 0.0
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

		if best_distance != float("inf") and second_best_distance != float("inf") and best_distance/second_best_distance < treshold:
			points_selected_1.append(points_1[i1])
			points_selected_2.append(points_2[best_index])
			dist_sum += best_distance
			dist_sum += second_best_distance

	if len(points_selected_1) != 0:
		dist_sum /= len(points_selected_1)
	else:
		dist_sum = float("inf")

	return points_selected_1, points_selected_2, dist_sum

def ssdPoint(pt1,pt2):
	return (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2

def getPoints(img_1, img_2):

	points_found_1 = computeCornerResponse(img_1,MIN_RADIUS_POINTS,NUM_POINTS_TO_FIND,HARRIS_TRESHOLD_RELATIVE)
	descriptors_1 = extractDescriptors(img_1,points_found_1,DESCRIPTOR_SHAPE_XY)

	points_found_2 = computeCornerResponse(img_2,MIN_RADIUS_POINTS,NUM_POINTS_TO_FIND,HARRIS_TRESHOLD_RELATIVE)
	descriptors_2 = extractDescriptors(img_2,points_found_2,DESCRIPTOR_SHAPE_XY)

	points_1, points_2, match_score = defineCorrespondences(descriptors_1,points_found_1,descriptors_2,points_found_2,RUSSIAN_GRANDMA_TRESHOLD)

	return match_score


if __name__ == '__main__':


	target_filename = 'tar.mp4'
	target_video_reader = imageio.get_reader(target_filename,  'mp4')
	target_fps = target_video_reader.get_meta_data()['fps']
	tar_num_frames = target_video_reader.get_meta_data()['nframes']
	print target_video_reader.get_meta_data()

	target_frames,target_indexes = getLowResolution(target_video_reader,INIT_TAR_FRAME,END_TAR_FRAME,0.1)

	saveFrames("out_low.mp4",target_frames,target_fps)
	
	self_probabilities, self_N = getProbabilities(target_frames,target_indexes, target_video_reader, target_fps)

	figure, axis = plt.subplots(ncols=1)
	axis.imshow(trip(self_probabilities/np.max(self_probabilities)), vmin=0, vmax=1, interpolation="nearest")
	plt.show(block=True)

	true_tar_frames = target_frames[self_N:len(target_frames)-self_N-1]

	true_tar_indexes = target_indexes[self_N:len(target_indexes)-self_N]

	reference_filename = 'ref.mp4'
	reference_video_reader = imageio.get_reader(reference_filename,  'mp4')
	reference_fps = reference_video_reader.get_meta_data()['fps']
	ref_num_frames = reference_video_reader.get_meta_data()['nframes']

	ref_frames,ref_indexes = getLowResolution(reference_video_reader,INIT_REF_FRAME,END_REF_FRAME,0.1)


	avg_ref_frames = np.average(np.array(ref_frames),axis=0)
	avg_true_tar_frames = np.average(np.array(true_tar_frames),axis=0)

	figure, axis = plt.subplots(ncols=1)
	axis.imshow(trip(avg_ref_frames), vmin=0, vmax=1, interpolation="nearest")
	plt.show(block=True)
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(trip(avg_true_tar_frames), vmin=0, vmax=1, interpolation="nearest")
	plt.show(block=True)

	for i,f in enumerate(ref_frames):
		s = f-avg_ref_frames
		t_in = np.where(s > 0.01)
		ref_frames[i] = f*0.0
		ref_frames[i][t_in] = 1.0

		if t_in[0].size != 0:
			current_0 = np.average(t_in[0])
			current_1 = np.average(t_in[1])

			desired_0 = f.shape[0]/2
			desired_1 = f.shape[1]/2

			ref_frames[i] = np.roll(ref_frames[i],int(desired_0-current_0),axis=0)
			ref_frames[i] = np.roll(ref_frames[i],int(desired_1-current_1),axis=1)

	for i,f in enumerate(true_tar_frames):
		s = f-avg_ref_frames
		t_in = np.where(s > 0.01)
		true_tar_frames[i] = f*0.0
		true_tar_frames[i][t_in] = 1.0

		if t_in[0].size != 0:
			current_0 = np.average(t_in[0])
			current_1 = np.average(t_in[1])

			desired_0 = f.shape[0]/2
			desired_1 = f.shape[1]/2

			true_tar_frames[i] = np.roll(true_tar_frames[i],int(desired_0-current_0),axis=0)
			true_tar_frames[i] = np.roll(true_tar_frames[i],int(desired_1-current_1),axis=1)

	saveFrames("mot_ref.mp4",ref_frames,reference_fps)
	saveFrames("mot_tar.mp4",true_tar_frames,target_fps)

	for i,f in enumerate(ref_frames):
		ref_frames[i] = rescale(f,0.3)
	for i,f in enumerate(true_tar_frames):
		true_tar_frames[i] = rescale(f,0.3)

	figure, axis = plt.subplots(ncols=1)
	axis.imshow(trip(ref_frames[3]), vmin=0, vmax=1, interpolation="nearest")
	plt.show(block=True)
	figure, axis = plt.subplots(ncols=1)
	axis.imshow(trip(true_tar_frames[3]), vmin=0, vmax=1, interpolation="nearest")
	plt.show(block=True)

	distance = np.zeros((len(ref_frames),len(true_tar_frames)))
	for ri, ref_frame in enumerate(ref_frames):
		for ti, tar_frame in enumerate(true_tar_frames):
			print (ri,ti)
			distance[ri,ti] = ssd(ref_frame,tar_frame)


	figure, axis = plt.subplots(ncols=1)
	axis.imshow(trip(distance), vmin=0, vmax=1, interpolation="nearest")
	plt.show(block=True)

	writer = imageio.get_writer("mimic_video.mp4", fps=reference_fps)
	low_writer = imageio.get_writer("low_mimic_video.mp4", fps=reference_fps)
	i = 0
	last_ti = -1
	for ri, ref in enumerate(ref_frames):
		i += 1
		print i
		min_dist = float("inf")
		min_index = None
		for ti, tar in enumerate(true_tar_frames):

			if last_ti == -1:
				d = distance[ri,ti]
			else:
				d = distance[ri,ti] * (1-self_probabilities[last_ti,ti])

			if d < min_dist:
				min_dist = d
				min_index = ti

		last_ti = min_index
		writer.append_data(np.concatenate((reference_video_reader.get_data(ref_indexes[ri]),target_video_reader.get_data(true_tar_indexes[min_index])),axis=1))
		low_writer.append_data(np.concatenate((ref,true_tar_frames[min_index]),axis=1))

	writer.close()
	low_writer.close()

