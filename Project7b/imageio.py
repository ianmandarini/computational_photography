import skimage.io as skio
from skimage import img_as_float
import glob
import sys

def getImageNames(folder):
	query = "%s/*" % folder
	results = glob.glob(query)
	num_results = len(results)

	if num_results == 0:
		sys.exit("Error: Images Not Found")
	if num_results < 2:
		sys.exit("Error: More images are needed!")
	
	return results

def readImageFloat(file_name):
	print "Reading file " + file_name
	image = skio.imread(file_name)
	return img_as_float(image)