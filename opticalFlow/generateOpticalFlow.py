#generate optical flow for images

import numpy as np 
import cv2 
import os

path_data = '../driving_dataset'
path_out = '../optical_flow_dataset'

#params for ShiTomasi corner detection
feature_params = dict(	maxCorners = 100,
						qualityLevel = 0.3,
						minDistance = 7,
						blockSize = 7)

#params for lucas kanade optical flow
lk_params = dict(	winSize = (15,15),
					maxLevel = 2,
					criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#rand. colors
color = np.random.randint(0, 255, (100,3))

#first frame
currentFrame = 0

#load image in grayscale
imgpath = os.path.join(path_data, str(currentFrame) + '.jpg')
#print imgpath
old_frame = cv2.imread(imgpath, 0)
currentFrame += 1

#print old_frame.shape

#decide which points are good for tracking via Shi-Tomasi
p0 = cv2.goodFeaturesToTrack(old_frame, mask = None, **feature_params)

#print p0.shape

#create mask img for drawing purposes
refImg = cv2.imread(imgpath)
mask = np.zeros_like(refImg)

totalImgs = 45568
while(currentFrame<totalImgs):
	#print currentFrame
	#load next img
	imgpath = os.path.join(path_data, str(currentFrame) + '.jpg')
	new_frame = cv2.imread(imgpath, 0)

	#print new_frame.shape

	#calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None, **lk_params)

	if st is None:
		#save optical flow from image before before
		cv2.imwrite(os.path.join(path_out, str(currentFrame) + '.jpg'), mask)	#img
		continue

	#select good points
	good_new = p1[st==1]
	good_old = p0[st==1]

	#print good_new.shape

	#draw tracks
	frame_rgb = cv2.imread(imgpath)
	for i,(new,old) in enumerate(zip(good_new,good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		#colorized mask:
		#cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		#white color mask:
		cv2.line(mask, (a,b),(c,d), [255,255,255], 2)
		#print mask.shape
		#cv2.circle(frame_rgb, (a,b), 5, color[i].tolist(), -1)
		#print frame_rgb.shape

	#add mask and image together
	#img = cv2.add(frame_rgb, mask)

	#show img
	#cv2.imshow('frame', img) #mask
	#cv2.waitKey(50)
	
	#write img out
	imgname_out = os.path.join(path_out, str(currentFrame) + '.jpg')
	#print imgname_out
	cv2.imwrite(imgname_out, mask) #img
	
	#update frames
	currentFrame += 1
	old_frame = new_frame.copy()
	p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()






