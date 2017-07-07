from __future__ import with_statement
import numpy as np
import tensorflow as tf
import  os

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


		#return trainX, valX, testX, trainY, valY, testY

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

if __name__ == "__main__":
	main()