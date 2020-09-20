import numpy as np
import pandas as pd
from collections import defaultdict


class Data():
	def __init__(self, args):
		self.tracks = []
		self.vehicle_tracks = defaultdict(list)

		track_info = pd.read_csv(args.vehicle_path, sep=';', header=0)
		track_info['Camera ID'] = track_info['Camera ID'].astype('int') + 1
		track_info['Time'] = pd.to_datetime(track_info['Time'])

		for line in open(args.feature_path, 'r'):
			track_path, reid_feature = line.strip().split(';')

			dir_list = track_path.split('/')
			track_id = dir_list[-1].split('.')[0]
			vehicle_id = dir_list[-2]
			vehicle_track = vehicle_id + '/' + track_id
			track_df = track_info[track_info['id/image'] == vehicle_track]

			track = dict()
			track['track_id'] = int(track_id)
			track['vehicle_id'] = int(vehicle_id)
			track['reid_feature'] = np.array(reid_feature.split(',')).astype(np.float32)
			track['camera_id'] = track_df['Camera ID'].values[0]
			track['time'] = track_df['Time'].values[0]

			self.tracks.append(track)
			self.vehicle_tracks[vehicle_id].append(track['track_id'])

		self.tracks.sort(key=lambda x: x['time'])
		print('total tracks:{}, total vehicles:{}'.format(len(self.tracks), len(self.vehicle_tracks)))
