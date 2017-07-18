from __future__ import with_statement
from PIL import Image as pimg
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
import  os
import preprocessing as pre
import logging
import time
from datetime import datetime

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 27
NUM_THREADS = 16
NUM_SAMPLES = 27243
NUM_BATCHES = 1009
MIN_QUEUE_SIZE = int(NUM_SAMPLES * 0.4)
NUM_ITER = 100000

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 100.0       # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1   # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.0001     # Initial learning rate.

Y_MEAN =  0.473634918528
U_MEAN = 0.0302694948176
V_MEAN = -0.0508748753295



def train():
	graph = tf.Graph()
	with graph.as_default():
		images, labels = read_data()

		# First convolutional layer 
		W_conv1 = weight_variable('conv_weights_1', [5, 5, 3, 24], 0.01)
		b_conv1 = bias_variable('conv_biases_1', [24])
		h_conv1 = tf.nn.relu(conv2d(images, W_conv1) + b_conv1)

		# Pooling layer - downsamples by 2X.
		max_pool_1 = max_pool_2x2(h_conv1)

		# Second convolutional layer 
		W_conv2 = weight_variable('conv_weights_2', [5, 5, 24, 36], 24.0)
		b_conv2 = bias_variable('conv_biases_2', [36])
		h_conv2 = tf.nn.relu(conv2d(max_pool_1, W_conv2) + b_conv2)

		# Second Pooling layer
		max_pool_2 = max_pool_2x2(h_conv2)

		# Third convolutional layer 
		W_conv3 = weight_variable('conv_weights_3', [5, 5, 36, 48], 36.0)
		b_conv3 = bias_variable('conv_biases_3', [48])
		h_conv3 = tf.nn.relu(conv2d(max_pool_2, W_conv3) + b_conv3)

		# Third Pooling layer
		max_pool_3 = max_pool_2x2(h_conv3)

		# Fourth convolutional layer 
		W_conv4 = weight_variable('conv_weights_4', [3, 3, 48, 64], 48.0)
		b_conv4 = bias_variable('conv_biases_4', [64])
		h_conv4 = tf.nn.relu(conv2d(max_pool_3, W_conv4) + b_conv4)

		# Fifth convolutional layer 
		W_conv5 = weight_variable('conv_weights_5', [3, 3, 64, 64], 64.0)
		b_conv5 = bias_variable('conv_biases_5', [64])
		h_conv5 = tf.nn.relu(conv2d(h_conv4, W_conv5) + b_conv5)

		#stack result into one dimensional vector by using -1 option
		conv_flat = tf.reshape(h_conv5, [BATCH_SIZE, -1]) 

		# Fully connected layer 1
		W_fc1 = weight_variable('fc_weights_1', [1 * 18 * 64, 1164], 1164.0)
		b_fc1 = bias_variable('fc_biases_1', [1164])
		h_fc1 = tf.nn.relu(tf.matmul(conv_flat, W_fc1) + b_fc1)

		# Fully connected layer 2
		W_fc2 = weight_variable('fc_weights_2', [1164, 100], 100.0)
		b_fc2 = bias_variable('fc_biases_2', [100])
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		# Fully connected layer 3
		W_fc3 = weight_variable('fc_weights_3', [100, 10], 10.0)
		b_fc3 = bias_variable('fc_biases_3', [10])
		h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

		# Fully connected layer 4
		W_fc4 = weight_variable('fc_weights_4', [10, 1], 1.0)
		b_fc4 = bias_variable('fc_biases_4', [1])
		h_fc4 = tf.matmul(h_fc3, W_fc4) + b_fc4

		#exponential learning rate decay
		global_step = tf.Variable(0, name='gloabal_step', trainable=False)
		decay_steps = int(NUM_BATCHES * NUM_EPOCHS_PER_DECAY)
		lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
										global_step,
										decay_steps,
										LEARNING_RATE_DECAY_FACTOR,
										staircase=True)
		tf.summary.scalar('learning_rate', lr)

		#loss and gradient computation
		loss,train_op = loss_func(h_fc4, labels, global_step, lr)
		tf.summary.scalar('loss', loss)

		# max_to_keep option to store all weights
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

		logging.basicConfig(filename='/work/raymond/dlcv/dlcv_visnav/src/log/training.log',level=logging.INFO)


		for x in range(NUM_ITER):
			average_loss = 0.0
			start_time = time.time()
			curr_learnRate = 0.0

			for y in range(NUM_BATCHES):
				#print("testing...")
				summary, train_out, lossVal, image_out, lr_out = session.run([merged, train_op, loss, images, lr])
				train_writer.add_summary(summary, x*NUM_ITER + y)
				print('iteration: ', x)
				print("learning_rate: ", lr_out)
				print('loss: ', lossVal)

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
			# 	#print("done")
				print('batch: ', y)
				# print(lossVal)
			# 	# print(image_out.shape)
				curr_learnRate = lr_out
				
			# 	#break
			
			average_loss = average_loss/NUM_BATCHES
			print("average_loss: ", average_loss)
			
			str1 = str(x)
			str2 = "check_files/model"
			str3 = ".ckpt"
			str4 = str2 + str1 + str3

			save_path = saver.save(session, str4, global_step=x)

			content = datetime.now(), x, curr_learnRate, average_loss
			logging.info(content)


		
		train_writer.close()
		#tensorflow threads 
		coord.request_stop()
		coord.join(threads)

def loss_func(logits, labels, global_step, learning_rate):
	loss = tf.reduce_mean((logits-labels)*(logits-labels))
	 # Compute the moving average of all individual losses and the total loss.
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.summary.scalar(l.op.name + '_raw_', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	# Compute gradients.
	with tf.control_dependencies([loss_averages_op]):
		opt = tf.train.AdamOptimizer(learning_rate)
		grads = opt.compute_gradients(loss)

	# Apply gradients.
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

	# Add histograms for trainable variables.
	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)

	# Add histograms for gradients.
	for grad, var in grads:
		if grad is not None:
			tf.summary.histogram(var.op.name + '/gradients', grad)

	# Track the moving averages of all trainable variables.
	variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	variables_averages_op = variable_averages.apply(tf.trainable_variables())

	with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
		train_op = tf.no_op(name='train')

	return loss,train_op

def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(name, shape, stddev):
	"""weight_variable generates a weight variable of a given shape."""
	initial = tf.truncated_normal_initializer(stddev=tf.rsqrt(stddev))
	return _variable_with_weight_decay(name, shape, initial, wd=0.0004)


def bias_variable(name, shape):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant_initializer(0.1)
	return _variable_with_weight_decay(name, shape, initial, wd=0.0004)

def _variable_on_cpu(name, shape, initializer):
	with tf.device('/cpu:0'):
		var = tf.get_variable(name, shape,tf.float32,initializer)
	return var

def _variable_with_weight_decay(name, shape, initializer, wd=None):
	"""
	Helper function to create an initialized variable with weight decay
	A weight decay act as a regularization, only is only added if specified.
	In our case, it is the l2-norm of the weights multiply by the weight decay value.
	"""

	var = _variable_on_cpu(name, shape, initializer)

	if wd is not None:
		weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
		tf.add_to_collection('losses', weight_decay)
	return var





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
	images, labels = tf.train.batch([yuv_image, label], batch_size=BATCH_SIZE,
		num_threads=NUM_THREADS, capacity=MIN_QUEUE_SIZE + 3 * BATCH_SIZE)

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
	train()

if __name__ == "__main__":
	main()