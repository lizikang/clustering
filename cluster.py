import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

import numpy as np
import searcher
import json
from collections import defaultdict
from util import *
from model.model import Model



class Cluster():
	def __init__(self, args):
		self.clusters = dict()
		self.tracks = []
		self.gallery = []

		self.vehicle_size = defaultdict(int)
		self.track_vehicle = dict()

		self.topk = args.topk
		self.reid_threshold = args.reid_threshold
		self.max_cluster_size = args.max_cluster_size

		if args.faiss_search_type == 'dot product':
			self.searcher = searcher.Searcher2(feat_len=256)
		elif args.faiss_search_type == 'euclid':
			self.searcher = searcher.Searcher(feat_len=256)

		with open(args.model_config_path, 'r') as f:
			args = json.load(fp=f)
			self.model = Model(args)
			self.model.restore()


	def add_cluster(self, track, cluster_id):
		centre_track = dict()
		centre_track['reid_feature'] = np.copy(track['reid_feature'])
		centre_track['reid_feature_sum'] = np.copy(track['reid_feature'])
		centre_track['camera_time'] = [[track['camera_id'], track['time']]]

		self.clusters[cluster_id] = dict()
		self.clusters[cluster_id]['centre_track'] = centre_track
		self.clusters[cluster_id]['size'] = 1
		self.clusters[cluster_id]['track_ids'] = [track['track_id']]


	def merge_cluster(self, track, cluster_id):
		reid_feature_sum = self.clusters[cluster_id]['centre_track']['reid_feature_sum'] + track['reid_feature']
		self.clusters[cluster_id]['centre_track']['reid_feature'] = feature_normalize(reid_feature_sum)
		self.clusters[cluster_id]['centre_track']['reid_feature_sum'] = reid_feature_sum
		self.clusters[cluster_id]['centre_track']['camera_time'].append([track['camera_id'], track['time']])

		self.clusters[cluster_id]['size'] += 1
		self.clusters[cluster_id]['track_ids'].append(track['track_id'])


	def add(self, track):
		track['reid_feature'] = feature_normalize(track['reid_feature'])
		self.track_vehicle[track['track_id']] = track['vehicle_id']
		cluster_nums = len(self.clusters)

		if cluster_nums == 0:
			self.add_cluster(track, 0)
			cluster_id = 0
		else:
			query = [track['reid_feature']]
			topk = min(self.topk, cluster_nums)
			top_scores, top_ids = self.searcher.search_by_topk(query, self.gallery, topk)
			best_reid_score, best_cluster_id, already_compared_cluster_ids, seq_score = -1e8, None, [], 0

			for i in top_ids[0]:
				cluster_id = self.tracks[i]['cluster_id']
				if cluster_id in already_compared_cluster_ids: continue
				already_compared_cluster_ids.append(cluster_id)
				if self.clusters[cluster_id]['size'] >= self.max_cluster_size: continue
				
				reid_score = np.inner(track['reid_feature'], self.clusters[cluster_id]['centre_track']['reid_feature'])
				if reid_score > best_reid_score:
					best_reid_score = reid_score
					best_cluster_id = cluster_id

			if best_cluster_id:
				camera_time_list = self.clusters[best_cluster_id]['centre_track']['camera_time']
				seq_score = self.get_seq_score(camera_time_list, track['camera_id']) 

			if best_reid_score >= self.reid_threshold or seq_score >= 0.98:
				cluster_id = best_cluster_id
				self.merge_cluster(track, cluster_id)
			else:
				cluster_id = cluster_nums
				self.add_cluster(track, cluster_id)

		track['cluster_id'] = cluster_id
		self.tracks.append(track)
		self.vehicle_size[track['vehicle_id']] += 1
		self.gallery.append(track['reid_feature'])
		self.searcher.add_feature([track['reid_feature']])
		print('track {} is in cluster {}'.format(track['track_id'], cluster_id))


	def evaluate(self):
		precisions, recalls = [], []
		for track in self.tracks:
			hits, cluster_id = 0, track['cluster_id']
			for track_id in self.clusters[cluster_id]['track_ids']:
				if track['vehicle_id'] == self.track_vehicle[track_id]:
					hits += 1

			precisions.append(hits / self.clusters[cluster_id]['size'])
			recalls.append(hits / self.vehicle_size[track['vehicle_id']])
		
		precision = np.mean(precisions)
		recall = np.mean(recalls)
		fscore = 2*precision*recall / (precision+recall)

		print('track numbers: {}, vehicle numbers: {}, cluster numbers: {}, track number per cluster: {:.2f}'.format
		     (len(self.tracks), len(self.vehicle_size), len(self.clusters), len(self.tracks)/len(self.clusters)))
		print('precision: {:.4f}, recall: {:.4f}, fscore: {:.4f}\n'.format
		     (precision, recall, fscore))


	def get_seq_score(self, camera_time_list, next_camera):
		camera_time_list.sort(key=lambda x: x[1])
		camera_ids = [i[0] for i in camera_time_list]
		camera_ids = get_recent(camera_ids, self.model.args['seq_len'])
		test_data = ([camera_ids], [[next_camera]])
		score = self.model.predict(test_data)
		return score[0][0]
