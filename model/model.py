import tensorflow as tf
import numpy as np
import json
import time
import math
from utils import *


class Model():
	def __init__(self, args):
		# define inputs
		self.input_seqs = tf.placeholder(tf.int32, [None, args['seq_len']])
		self.input_targets = tf.placeholder(tf.int32, [None, None])
		self.is_training = tf.placeholder(tf.bool, shape=[])
		self.dropout_prob = tf.cond(self.is_training, lambda: args['dropout_prob'], lambda: 0.0)
		self.args = args

		# embedding layer
		with tf.variable_scope('embedding_layer'):
			embedding_table = tf.get_variable(name='embedding_table', 
							  shape=[args['camera_size'], args['hidden_units']], 
							  initializer=tf.truncated_normal_initializer(stddev=0.01))
			embedding_table = tf.concat([tf.zeros(shape=[1, args['hidden_units']]), 
						     embedding_table[1:, :]], 0)
			input_seqs = tf.nn.embedding_lookup(embedding_table, self.input_seqs)

		# lstm layer
		with tf.variable_scope('lstm_layer'):
			cell_list = []
			for _ in range(args['num_layers']):
				cell_list.append(tf.nn.rnn_cell.BasicLSTMCell(args['hidden_units']))
			stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cell_list)

			sequence_length = tf.reduce_sum(tf.to_int32(tf.not_equal(self.input_seqs, 0)), 1)
			outputs, states = tf.nn.dynamic_rnn(stacked_cell, 
							    input_seqs, 
							    sequence_length=sequence_length, 
							    dtype=tf.float32)
			output = states[-1].h

		# fc layer
		with tf.variable_scope('fc_layer'):
			output = tf.nn.dropout(output, 1-self.dropout_prob)
			target_embs = tf.nn.embedding_lookup(embedding_table, self.input_targets)
			logits = tf.squeeze(tf.matmul(target_embs, tf.expand_dims(output, -1)), -1)
			self.preds = tf.nn.sigmoid(logits)

			pos_logits, neg_logits = logits[:, :1], logits[:, 1:]
			pos_loss = tf.reduce_mean(-tf.log(tf.sigmoid(pos_logits) + 1e-8))
			neg_loss = tf.reduce_mean(-tf.log(1-tf.sigmoid(neg_logits) + 1e-8))
			self.loss = pos_loss + neg_loss
			self.train_op = tf.train.AdamOptimizer(args['learn_rate']).minimize(self.loss)


	def train(self, train_data, valid_data):
		train_seqs, train_targets = train_data
		valid_seqs, valid_targets = valid_data

		train_bs = self.args['batch_size']
		train_iters = int(math.ceil(len(train_seqs) / train_bs))
		valid_bs = train_bs * 10
		valid_iters = int(math.ceil(len(valid_seqs) / valid_bs))

		with tf.Session() as sess:
			bacc, bepoch, times = 0, 0, []
			saver = tf.train.Saver(max_to_keep=1)
			sess.run(tf.global_variables_initializer())

			for i in range(1, self.args['num_epochs']+1):
				# train
				shuffled_indices = np.random.permutation(len(train_seqs))
				train_seqs = np.array(train_seqs)[shuffled_indices]
				train_targets = np.array(train_targets)[shuffled_indices]

				t0 = time.time()
				for j in range(1, train_iters+1):
					start, end = train_bs * (j-1), train_bs * j
					feed_dict = {self.input_seqs: train_seqs[start:end], 
						     self.input_targets: train_targets[start:end],
						     self.is_training: True}

					preds, loss, train_op = sess.run([self.preds, self.loss, self.train_op], 
									  feed_dict=feed_dict)
					if j % (train_iters//3) == 0: 
						print('epoch:{}, iteration:{}, loss:{:.4f}, accuracy:{:.2f}'.format
						     (i, j, loss, cal_acc(preds)))
				times.append(time.time() - t0)

				if i % 1 == 0:
					# valid
					preds_list = []
					for j in range(1, valid_iters+1):
						start, end = valid_bs * (j-1), valid_bs * j
						feed_dict = {self.input_seqs: valid_seqs[start:end], 
							     self.input_targets: valid_targets[start:end],
							     self.is_training: False}

						preds = sess.run(self.preds, feed_dict=feed_dict)
						preds_list.append(preds)
					acc = cal_acc(np.concatenate(preds_list))
					print('epoch:{}, valid accuracy:{:.2f}\n'.format(i, acc))

					# early stopping
					if acc >= bacc:
						bacc, bepoch = acc, i
						saver.save(sess, self.args['model_path']+'/model.ckpt', global_step=i)
					if i - bepoch >= 20:
						convergence_time = np.sum(times[:bepoch]) / 60
						epoch_time = np.mean(times) / 60
						print(self.args)
						print('epoch:{}, convergence time:{:.2f}min, epoch time:{:.2f}min, accuracy:{:.2f}\n'.format
						     (bepoch, convergence_time, epoch_time, bacc))

						with open(self.args['model_path']+'/config.json', 'w') as f:
							json.dump(self.args, f)
						break
						

	def inference(self, test_data):
		self.restore()
		self.predict(test_data)


	def restore(self):
		self.sess = tf.Session()
		saver = tf.train.Saver(max_to_keep=1)
		ckpt = tf.train.latest_checkpoint(self.args['model_path'])
		if ckpt:
			print("restoring model parameters from %s" % ckpt)
			saver.restore(self.sess, ckpt)
		else:
			print('there are no ckpt files in {}, please train first!'.format(self.args['model_path']))


	def predict(self, test_data):
		test_seqs, test_targets = test_data
		test_bs = self.args['batch_size'] * 10
		test_iters = int(math.ceil(len(test_seqs) / test_bs))

		times, preds_list = [], []
		for j in range(1, test_iters+1):
			start, end = test_bs * (j-1), test_bs * j
			feed_dict = {self.input_seqs: test_seqs[start:end], 
				     self.input_targets: test_targets[start:end],
				     self.is_training: False}

			t0 = time.time()
			preds = self.sess.run(self.preds, feed_dict=feed_dict)
			times.append(time.time()-t0)	
			preds_list.append(preds)

		inference_time = np.mean(times) * 1000
		preds = np.concatenate(preds_list)
		acc = cal_acc(preds)
		print('inference time:{:.2f}ms, accuracy:{:.2f}'.format(inference_time, acc))
		return preds
