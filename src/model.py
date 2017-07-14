from __future__ import with_statement
from PIL import Image as pimg
import numpy as np
import tensorflow as tf
import  os
import preprocessing as pre

BATCH_SIZE = 27
NUM_THREADS = 16
NUM_SAMPLES = 27243
NUM_BATCHES = 1009
MIN_QUEUE_SIZE = int(NUM_SAMPLES * 0.4)
NUM_ITER = 10000


NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1 # Initial learning rate.

def train():
	graph = tf.Graph()
	with graph.as_default():
		images, labels = read_data()

		#batch normalization
		images_norm = tf.layers.batch_normalization(images)

		#convolutional layers 
			# add regularization
		reg1 = tf.contrib.layers.l2_regularizer(scale= 0.0004)
		conv2d_1 = tf.layers.conv2d(inputs=images_norm, filters=24, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0), 
			kernel_regularizer=reg1)

		max_pool_1 = tf.layers.max_pooling2d(inputs=conv2d_1, pool_size=2, strides=2)

		reg2 = tf.contrib.layers.l2_regularizer(scale= 0.0004)
		conv2d_2 = tf.layers.conv2d(inputs=max_pool_1, filters=36, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0), 
			kernel_regularizer=reg2)

		max_pool_2 = tf.layers.max_pooling2d(inputs=conv2d_2, pool_size=2, strides=2, padding='same')

		reg3 = tf.contrib.layers.l2_regularizer(scale= 0.0004)
		conv2d_3 = tf.layers.conv2d(inputs=max_pool_2, filters=48, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0), 
			kernel_regularizer=reg3)

		max_pool_3 = tf.layers.max_pooling2d(inputs=conv2d_3, pool_size=2, strides=2, padding='same')

		reg4 = tf.contrib.layers.l2_regularizer(scale= 0.0004)
		conv2d_4 = tf.layers.conv2d(inputs=max_pool_3, filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0), 
			kernel_regularizer=reg4)

		reg5 = tf.contrib.layers.l2_regularizer(scale= 0.0004)
		conv2d_5 = tf.layers.conv2d(inputs=conv2d_4, filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0), 
			kernel_regularizer=reg5)

		#stack result into one dimensional vector by using -1 option
		conv_flat = tf.reshape(conv2d_5, [BATCH_SIZE, -1]) 

		#fully connected layers
		fc_1 = tf.contrib.layers.fully_connected(inputs=conv_flat, num_outputs=1164)

		fc_2 = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=100)

		fc_3 = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=50)

		fc_4 = tf.contrib.layers.fully_connected(inputs=fc_3, num_outputs=10)

		fc_5 = tf.contrib.layers.fully_connected(inputs=fc_4, num_outputs=1)

		#loss and gradient computation
		loss = loss_func(fc_5, labels)				
		tf.summary.scalar('loss', loss)


		#exponential learning rate decay
		global_step = tf.Variable(0, name='gloabal_step', trainable=False)
		decay_steps = int(NUM_BATCHES * NUM_EPOCHS_PER_DECAY)
  		lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
		tf.summary.scalar('learning_rate', lr)


		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		train_op = optimizer.minimize(loss, global_step=global_step)

		#max_to_keep option to store all weights
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

		


		#tensorflow session 
		session = tf.Session()

		#tensorboard
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('train', session.graph)

		#initialization of all variables
		session.run(tf.global_variables_initializer())
		session.run(tf.local_variables_initializer())

		#threads
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess=session)

		#save weights in directory
		#TODO file is empty
		#ckpt = tf.train.get_checkpoint_state('./weights/')





		for x in range(NUM_ITER):
			average_loss = 0.0
			for y in range(NUM_BATCHES):
				#print("testing...")
				summary, train_out, lossVal, image_out, lr_out= session.run([merged, train_op, loss, images, lr])
				train_writer.add_summary(summary, x*NUM_ITER + y)
				print("earning_rate: ", lr_out)
				
				# #test layers for output
				#iout, lout = session.run([fc_5, labels])
				#print(lout)

				# #------save image for verification-----
				# images = tf.cast(images, tf.uint8)
				# yuv = session.run(images)
				# if x == 0 and y <= 3:
				# 	newImg0 = pimg.fromarray(yuv[:,:,0])
				# 	newImg1 = pimg.fromarray(yuv[:,:,1])
				# 	newImg2 = pimg.fromarray(yuv[:,:,2])
				# 	str1 = "img"
				# 	str2 = str(y)
				# 	str3 = ".png"
				# 	print(str1+str2+str3)
				# 	newImg0.save(str1+"y"+str2+str3, "PNG")
				# 	newImg1.save(str1+"u"+str2+str3, "PNG")
				# 	newImg2.save(str1+"v"+str2+str3, "PNG")

				#if y > 3:
				#	break
				# #---------		

				average_loss = average_loss+lossVal
				#print("done")
				print('batch: ', y)
				print(lossVal)
				print(image_out.shape)
				
				#break
			
			average_loss = tf.divide(average_loss, BATCH_SIZE)
			print("avergae_loss: ", average_loss)
			
			str1 = str(x)
			str2 = "check_files/model"
			str3 = ".ckpt"
			str4 = str2 + str1 + str3

			save_path = saver.save(session, str4)

		
		train_writer.close()
		#tensorflow threads 
		coord.request_stop()
		coord.join(threads)

def loss_func(logits, labels):
	return tf.nn.l2_loss(tf.subtract(logits, labels))



def read_data():
	#convert data to tensor
	trainImages = tf.convert_to_tensor(pre.get_trainX());
	trainLables = tf.convert_to_tensor(pre.get_trainY());

	#create queue
	input_queue = tf.train.slice_input_producer([trainImages, trainLables], shuffle=True)
	image, label = get_data_from_disk(input_queue)

	#cast image to float
	image = tf.cast(image, tf.float32)
	label = tf.cast(label, tf.float32)

	#convert image from RGB to YUV
	conv_matrix = [[0.299, 0.589, 0.114], [-0.14713, -0.28886, 0.436],[0.615, -0.51499, -0.10001]]


	stacked_image = tf.reshape(image, [tf.shape(image)[0]*tf.shape(image)[1], tf.shape(image)[2]])
	stacked_image = tf.transpose(stacked_image)
	stacked_yuv = tf.matmul(conv_matrix, stacked_image)
	stacked_yuv = tf.transpose(stacked_yuv)
	yuv_image = tf.reshape(stacked_yuv, [tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]])




	#shapes have to be defined for mini batch creation
	yuv_image.set_shape([256, 455, 3])
	label.set_shape([])
	#mini batch
	images, labels = tf.train.batch([yuv_image, label], batch_size=BATCH_SIZE,
		num_threads=NUM_THREADS, capacity=MIN_QUEUE_SIZE + 3 * BATCH_SIZE)

	#crop image
	images = tf.image.resize_bilinear(images, [66,200])

	return images, labels



def get_data_from_disk(queue):
	imageFile = queue[0]
	content = tf.read_file(imageFile)
	image = tf.image.decode_jpeg(content, channels=3)

	label = queue[1]

	return image, label


def main():
	train()

if __name__ == "__main__":
	main()