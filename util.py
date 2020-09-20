import numpy as np


def feature_normalize(feature):
	eps = 1e-12
	return feature / (np.linalg.norm(feature) + eps)


def get_recent(sequence, k):
	sequence_length = len(sequence)
	if sequence_length >= k:
		return sequence[-k:]
	else:
		return list(np.pad(sequence, (0, k-sequence_length), 'constant'))
