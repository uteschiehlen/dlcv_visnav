from __future__ import with_statement
from PIL import Image as pimg
import numpy as np
import  os
import math
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd

trainX = None
trainY = None
valX  = None
valY  = None
testX  = None
testY = None

trainX_balanced=None
trainY_balanced=None

NUM_THREADS = 16

Y_MEAN =  0.473634918528
U_MEAN = 0.0302694948176
V_MEAN = -0.0508748753295


def preprocessing():
	global trainX 
	global valX 
	global testX 

	global trainY 
	global valY 
	global testY 
	filenames = []
	angles = []
	
	#dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../driving_dataset/data.txt')
	#fileh = open(dir)
	#try:
	with open('../driving_dataset/data.txt') as f:
		content = f.readlines()
		content= [x.split() for x in content]
		
		#read data from file		
		for x in content:
			filenames.append('../driving_dataset/' + x[0])
			angles.append(x[1])


		filenamesArray = np.array(filenames)
		anglesArray = np.array(angles, dtype=np.float32)

		#convert angles to radians
		anglesArray = (anglesArray*np.pi)/180

		#split data into 60% 20% 20%
		trainX = filenamesArray[0: np.floor(filenamesArray.size *0.65).astype(int)]
		valX = filenamesArray[np.floor(filenamesArray.size *0.65).astype(int): np.floor(filenamesArray.size *0.8).astype(int)]
		testX = filenamesArray[np.floor(filenamesArray.size *0.8).astype(int): filenamesArray.size ]

		trainY = anglesArray[0: np.floor(anglesArray.size *0.65).astype(int)]
		valY = anglesArray[np.floor(anglesArray.size *0.65).astype(int): np.floor(anglesArray.size *0.8).astype(int)]
		testY = anglesArray[np.floor(anglesArray.size *0.8).astype(int): anglesArray.size ]

		#for GPU handling
		np.savetxt('trainX.txt',  trainX, fmt="%s")
		np.savetxt('valX.txt', valX, fmt="%s")
		np.savetxt('testX.txt', testX, fmt="%s")
		np.savetxt('trainY.txt', trainY, fmt="%s")
		np.savetxt('valY.txt', valY, fmt="%s")
		np.savetxt('testY.txt', testY, fmt="%s")

		train = np.column_stack((trainX, trainY))
		np.savetxt('train.txt',  train, fmt="%s")

		val = np.column_stack((valX, valY))
		np.savetxt('val.txt',  val, fmt="%s")

		test = np.column_stack((testX, testY))
		np.savetxt('test.txt',  test, fmt="%s")

def generate_balance_dataset():
	create_balance_dataset('train.txt', 'train')
	# plot_balance_dataset('../driving_dataset/train_balanced.csv')


def plot_balance_dataset(file):
	# temp = pd.read_csv('../driving_datas et/driving_log_balanced.csv',header=None, names=['image, steering'])
	temp = pd.read_csv(file, sep=" ",header=None, names=['image', 'steering'])
	temp.steering = temp.steering.astype(float)
	# print(temp)

	plt.hist(np.absolute(temp.steering), bins=1000)  # arguments are passed to np.histogram
	plt.title("Histogram with 1000 bins")
	plt.show()

def create_balance_dataset(file, mode):

	global trainX_balanced, trainY_balanced

	data = pd.read_csv(file, sep=" ", header = None, names=['image', 'steering'])

	minVal = 0.0
	maxVal = 501.78

	balanced = pd.DataFrame()
	bins = 1000
	bin_n = 200
	start = minVal

	count = 1
	for end in np.linspace(minVal, maxVal, num=bins):  
		data_range = data[(np.absolute(data.steering) >= start) & (np.absolute(data.steering) < end)]
		range_n = min(bin_n, data_range.shape[0])
		
		if range_n > 0 :
			balanced = pd.concat([balanced, data_range.sample(range_n)])
			# print('count: ', count)
			count += 1
		start = end
	balanced.to_csv('../driving_dataset/' +mode+'_balanced.csv', index=False, header=False, sep=" ")
	
	temp = balanced.as_matrix()

	trainX_balanced = temp[:,0]
	trainY_balanced = temp[:,1].astype(np.float32)

	
	

def show_histrogram(data):

	# print(np.amin(trainY_positive))
	# print(np.amax(trainY_positive))

	minVal = np.amin(data)
	maxVal = np.amax(data)

	#number of bins to be the half of the range of values
	bins = math.floor((maxVal-minVal)*2)

	plt.hist(data, bins=bins)  # arguments are passed to np.histogram
	plt.title("Histogram with " + str(bins)+" bins")
	plt.show()

def get_train():

	global trainX_balanced
	global trainY_balanced
	if trainX_balanced is None and trainY_balanced is None:
		preprocessing()
		generate_balance_dataset()
		
	return  trainX_balanced, trainY_balanced


def get_val():
	global valX
	global valY
	if valX is None or valY is None:
		preprocessing()
	return valX, valY

def get_test():
	global testX
	global testY
	if testX is None or testY is None:
		preprocessing()
	return testX, testY



def read_data(inputImages, inputLabels, batch_size, num_samples, shuffle):
	
	#convert data to tensor
	inputImages = tf.convert_to_tensor(inputImages)
	inputLabels = tf.convert_to_tensor(inputLabels)


	#create queue
	input_queue = tf.train.slice_input_producer([inputImages, inputLabels], shuffle=shuffle)
	image, label = get_data_from_disk(input_queue)

	# apply random horizontal flip
	choice = tf.random_uniform(shape=[1], minval=0, maxval=2, dtype=tf.int32)
	if shuffle:
		image = tf.cond(tf.equal(tf.squeeze(choice), 0), lambda:tf.image.flip_left_right(image), lambda:image)
		label = tf.cond(tf.equal(tf.squeeze(choice), 0), lambda:-label, lambda:label)

	#cast image to float
	image = tf.cast(image, tf.float32)
	label = tf.cast(label, tf.float32)

	#normalization to [0,1]
	image = image/255

	#convert image from RGB to YUV
	conv_matrix = [[0.299, 0.589, 0.114], [-0.14713, -0.28886, 0.436],[0.615, -0.51499, -0.10001]]


	stacked_image = tf.reshape(image, [tf.shape(image)[0]*tf.shape(image)[1], tf.shape(image)[2]])
	stacked_image = tf.transpose(stacked_image)
	stacked_yuv = tf.matmul(conv_matrix, stacked_image)
	stacked_yuv = tf.transpose(stacked_yuv)
	yuv_image = tf.reshape(stacked_yuv, [tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]])

	mean = tf.constant([Y_MEAN, U_MEAN, V_MEAN], dtype=tf.float32, shape=[1, 1, 3], name='yuv_mean')
	yuv_image = yuv_image-mean

	#shapes have to be defined for mini batch creation
	yuv_image.set_shape([256, 455, 3])
	label.set_shape([])
	#mini batch
	images, labels = tf.train.batch([yuv_image, label], batch_size=batch_size,
		num_threads=NUM_THREADS, capacity=int(num_samples * 0.4) + 3 * batch_size)

	#crop image
	images = tf.image.resize_bilinear(images, [66,200])

	return images, labels


def get_data_from_disk(queue):
	imageFile = queue[0]

	# imageFile = "../driving_dataset/24.jpg"
	content = tf.read_file(imageFile)
	image = tf.image.decode_jpeg(content, channels=3)

	label = queue[1]
	# label = 11.7

	return image, label



def main():
	preprocessing()
	generate_balance_dataset()

if __name__ == "__main__":
	main()