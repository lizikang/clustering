import pandas as pd
import numpy as np
import random
from utils import *


class Data():
	def __init__(self, args):
		# read data file
		df = pd.read_csv(args['data_path'], sep=';', header=0)
		df['vehicle_id'] = df['id/image'].apply(lambda x: x.strip().split('/')[0])
		df = df.sort_values(['vehicle_id', 'Time'])
		
		df['Camera ID'] = df['Camera ID'] + 1
		camera_set = set(df['Camera ID'])
		self.camera_size = len(camera_set) + 1

		# split dataset
		train_set, valid_set, test_set = {}, {}, {}
		vehicle_list, camera_list = list(df['vehicle_id']), list(df['Camera ID'])
		vehicle_ids, indices, counts = np.unique(vehicle_list, return_index=True, return_counts=True)

		for i in range(len(vehicle_ids)):
			vid, index, length = vehicle_ids[i], indices[i], counts[i]
			cameras = camera_list[index: index+length]
			train_set[vid], valid_set[vid], test_set[vid] = cameras[:-2], [cameras[-2]], [cameras[-1]]

		# generate input data
		train_seqs, train_targets = [], []
		valid_seqs, valid_targets = [], []
		test_seqs, test_targets = [], []
		random.seed(11)

		for vid in vehicle_ids:
			rated = train_set[vid] + valid_set[vid] + test_set[vid]
			sample_pool = camera_set - set(rated)

			# train
			for seq in slide_window(train_set[vid], args['seq_len']+1):
				train_seqs.append(seq[:-1])
				train_targets.append(seq[-1:] + random.sample(sample_pool, args['sample_ratio']))

			# valid
			valid_seqs.append(get_recent(train_set[vid], args['seq_len']))
			valid_targets.append(valid_set[vid] + random.sample(sample_pool, args['sample_ratio']))

			# test
			test_seqs.append(get_recent(train_set[vid], args['seq_len']-1) + valid_set[vid])
			test_targets.append(test_set[vid] + random.sample(sample_pool, args['sample_ratio']))

		self.train_data = (train_seqs, train_targets)
		self.valid_data = (valid_seqs, valid_targets)
		self.test_data = (test_seqs, test_targets)
