#generate txt file

import os

path_old = '../driving_dataset'
path_new = '../driving_dataset_long'

filenames = []
angles = []
lenLongestFileName = 5


with open(os.path.join(path_old, 'data.txt')) as f_in:
	with open(os.path.join(path_new, 'data.txt'), 'w') as f_out:
		content = f_in.readlines()
		content = [x.split() for x in content]

		for x in content:
			#create new filename
			fileN = x[0].split('.')
			lenFile = len(fileN[0])
			newFilename = x[0]

			if lenFile < lenLongestFileName:
				diff = lenLongestFileName - lenFile
				newFilename = '0' * diff
				newFilename += x[0]

			#append new filename and respective angle
			filenames.append(newFilename)
			angles.append(x[1])

			#write to file
			newline = newFilename + ' ' + x[1] + '\n'
			f_out.write(newline)
