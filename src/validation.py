from __future__ import with_statement
from PIL import Image as pimg
import numpy as np
import tensorflow as tf
import  os
import preprocessing as pre
import logging
import time
from datetime import datetime


tf.logging.set_verbosity(tf.logging.INFO)

# BATCH_SIZE = 44
BATCH_SIZE = 139
NUM_THREADS = 16
NUM_SAMPLES = 6811
NUM_BATCHES = int(NUM_SAMPLES/BATCH_SIZE)
MIN_QUEUE_SIZE = int(NUM_SAMPLES * 0.4)

def validate():
	graph = tf.Graph()
	with graph.as_default():

		global_step = tf.Variable(0, name='global_step', trainable=False)

		im, la = pre.get_val()
		images, labels = pre.read_data(im, la, BATCH_SIZE, NUM_SAMPLES, False)

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

		# radiants in the range of [-pi/2, pi/2] * 2 to get 360 range
		y = tf.multiply(tf.atan(h_fc4), 2)

		saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)


		#tensorflow session 
		session = tf.Session()

		#initialization of all variables
		session.run(tf.global_variables_initializer())
		session.run(tf.local_variables_initializer())

		#threads
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess=session)

		#save weights in directory
		#TODO file is empty
		# ckpt = tf.train.get_checkpoint_state('./weights/')


		logging.basicConfig(filename='../log/validation4.log',level=logging.INFO)

		accs = []
		for i in range(201):
			accuracy = 0.0
			saver.restore(session, '../weights/model' + str(i) + '.ckpt-' + str(i))

			for b in range(NUM_BATCHES):
			
				y_out,  image_out, label_out = session.run([y, images, labels])
				#print('epoche ' + str(i) + ': ' + str(y_out) + '-' + str(label_out))
				batch_acc = comp_accuracy(y_out, label_out)	
				accuracy += batch_acc
				#print('batch accuracy: ',batch_acc)

			accuracy = accuracy/NUM_BATCHES
			accs.append(accuracy)
			#print('epoch ' + str(i) + ' accuracy: ', accuracy)

			content = datetime.now(), i, accuracy
			logging.info(content)

		print(np.argmin(accs), np.min (accs))

		#tensorflow threads 
		coord.request_stop()
		coord.join(threads)


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(name, shape, stddev):
	"""weight_variable generates a weight variable of a given shape."""
	#1/sqrt(x)
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
	return var

def comp_accuracy(predictions, labels):
	return ((predictions-labels)**2).mean(axis=None)
	#loss = tf.reduce_mean(tf.squared_difference(predictions, labels))



def main():
	validate()

if __name__ == "__main__":
	main()