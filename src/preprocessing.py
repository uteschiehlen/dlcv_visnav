from __future__ import with_statement
import numpy as np
import  os
import math

import matplotlib.pyplot as plt
import pandas as pd

trainX = None
trainY = None
valX  = None
valY  = None
testX  = None
testY = None


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

		#split data into 60% 20% 20%
		trainX = filenamesArray[0: np.floor(filenamesArray.size *0.6).astype(int)]
		valX = filenamesArray[np.floor(filenamesArray.size *0.6).astype(int): np.floor(filenamesArray.size *0.8).astype(int)]
		testX = filenamesArray[np.floor(filenamesArray.size *0.8).astype(int): filenamesArray.size ]

		trainY = anglesArray[0: np.floor(anglesArray.size *0.6).astype(int)]
		valY = anglesArray[np.floor(anglesArray.size *0.6).astype(int): np.floor(anglesArray.size *0.8).astype(int)]
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
	create_balance_dataset('train.txt')
	plot_balance_dataset('../driving_dataset/train_balanced.csv')


def plot_balance_dataset(file):
	# temp = pd.read_csv('../driving_datas et/driving_log_balanced.csv',header=None, names=['image, steering'])
	temp = pd.read_csv(file, sep=",",header=None, names=['image', 'steering'])
	temp.steering = temp.steering.astype(float)
	# print(temp)

	plt.hist(np.absolute(temp.steering), bins=1000)  # arguments are passed to np.histogram
	plt.title("Histogram with 1000 bins")
	plt.show()

def create_balance_dataset(file):
	# data = pd.read_csv('../driving_dataset/data.txt', sep=" ", header = None, names=['image', 'steering'])
	# print(data)

	data = pd.read_csv(file, sep=" ", header = None, names=['image', 'steering'])

	minVal = 0.0
	maxVal = 501.78

	balanced = pd.DataFrame()
	bins = 1000
	bin_n = 75
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
	balanced.to_csv('../driving_dataset/train_balanced.csv', index=False, header=False)
		

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


def get_trainX():
	global trainX
	if  trainX is None:
		preprocessing()
	return  trainX

def get_trainY():
	global trainY
	if  trainY is None:
		preprocessing()
	return  trainY


def main():
	preprocessing()
	generate_balance_dataset()

if __name__ == "__main__":
	main()