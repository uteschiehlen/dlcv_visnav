import os
path = '../driving_dataset/'
newpath = '.'
for filename in os.listdir(path):

	num = filename[:-4]
	num = num.zfill(8)
	new_filename = 'frame_' + num + ".jpg"
	os.rename(os.path.join(path, filename), os.path.join(newpath, new_filename))
	print(os.path.join(newpath, new_filename))
	# print(filename)

print("done")