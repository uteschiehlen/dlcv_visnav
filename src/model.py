from __future__ import with_statement
import numpy as np
import tensorflow as tf
import  os
import preprocessing as pre

batch_size = 9
num_threads = 16
num_samples = 27243
num_batches = 3027
min_queue_size = int(num_samples * 0.4)
num_iter = 10000

def train():
	graph = tf.Graph()
	with graph.as_default():
		images, labels = read_data()

		#batch normalization
		images_norm = tf.layers.batch_normalization(images)

		#convolutional layers
		conv2d_1 = tf.layers.conv2d(inputs=images_norm, filters=24, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0))

		max_pool_1 = tf.layers.max_pooling2d(inputs=conv2d_1, pool_size=2, strides=2)

		conv2d_2 = tf.layers.conv2d(inputs=max_pool_1, filters=36, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0))

		max_pool_2 = tf.layers.max_pooling2d(inputs=conv2d_2, pool_size=2, strides=2, padding='same')

		conv2d_3 = tf.layers.conv2d(inputs=max_pool_2, filters=48, kernel_size=5, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0))

		max_pool_3 = tf.layers.max_pooling2d(inputs=conv2d_3, pool_size=2, strides=2, padding='same')

		conv2d_4 = tf.layers.conv2d(inputs=max_pool_3, filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0))

		conv2d_5 = tf.layers.conv2d(inputs=conv2d_4, filters=64, kernel_size=3, strides=1, padding='valid', activation=tf.nn.relu, 
			kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32), bias_initializer=tf.constant_initializer(0.0))

		#stack result into one dimensional vector by using -1 option
		conv_flat = tf.reshape(conv2d_5, [batch_size, -1]) 

		#fully connected layers
		fc_1 = tf.contrib.layers.fully_connected(inputs=conv_flat, num_outputs=1164)

		fc_2 = tf.contrib.layers.fully_connected(inputs=fc_1, num_outputs=100)

		fc_3 = tf.contrib.layers.fully_connected(inputs=fc_2, num_outputs=50)

		fc_4 = tf.contrib.layers.fully_connected(inputs=fc_3, num_outputs=10)

		fc_5 = tf.contrib.layers.fully_connected(inputs=fc_4, num_outputs=1)

		#loss and gradient computation
		loss = loss_func(fc_5, labels)
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		global_step = tf.Variable(0, name='gloabal_step', trainable=False)
		train_op = optimizer.minimize(loss, global_step=global_step)

		#max_to_keep option to store all weights
		saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

		#tensorflow session and threads
		session = tf.Session()
		session.run(tf.global_variables_initializer())
		session.run(tf.local_variables_initializer())

		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord, sess=session)

		#save weights in directory
		ckpt = tf.train.get_checkpoint_state('./weights/')



		for x in range(num_iter):
			for y in range(num_batches):
				#print("testing...")
				_, lossVal = session.run([train_op, loss])
				#iout, lout = session.run([fc_5, labels])
				#print("done")
				print(lossVal)
				#print(lout)
				#break

			
		
		#tensorflow threads 
		coord.request_stop()
		coord.join(threads)

def loss_func(logits, labels):
	return tf.nn.l2_loss(tf.square(tf.subtract(logits, labels)))



def read_data():
	#convert data to tensor
	trainImages = tf.convert_to_tensor(pre.get_trainX());
	trainLables = tf.convert_to_tensor(pre.get_trainY());

	#create queue
	input_queue = tf.train.slice_input_producer([trainImages, trainLables], shuffle=False)
	image, label = get_data_from_disk(input_queue)

	#cast image to float
	image = tf.cast(image, tf.float32)
	label = tf.cast(label, tf.float32)


	#shapes have to be defined for mini batch creation
	image.set_shape([256, 455, 3])
	label.set_shape([])
	#mini batch
	images, labels = tf.train.batch([image, label], batch_size=batch_size,
		num_threads=num_threads, capacity=min_queue_size + 3 * batch_size)

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