#adapt images

import cv2
import os

path_old = '../driving_dataset'
path_new = '../driving_dataset_long'

lenLongestFileName = 5

for filename in os.listdir(path_old):
	img = cv2.imread(os.path.join(path_old, filename))
	if img is not None:
		fileN = filename.split('.')
		lenFile = len(fileN[0])
		newFilename = filename

		if lenFile < lenLongestFileName:
			diff = lenLongestFileName - lenFile
			newFilename = '0' * diff
			newFilename += filename
		
		cv2.imwrite(os.path.join(path_new, newFilename), img)