import numpy as np


def slide_window(sequence, window_size, step_size=1):
	sequence_length = len(sequence)

	if sequence_length >= window_size:
		for i in range(0, sequence_length-window_size+1, step_size):
				yield sequence[i: i+window_size]
	else:
		padding_size = window_size - sequence_length
		yield list(np.pad(sequence, (padding_size, 0), 'constant'))


def get_recent(sequence, k):
	sequence_length = len(sequence)
	if sequence_length >= k:
		return sequence[-k:]
	else:
		return list(np.pad(sequence, (0, k-sequence_length), 'constant'))


def cal_acc(preds, threshold=0.5):
	pos_preds = preds[:,:1]
	neg_preds = preds[:,1:]
	pos_hits = np.int32(pos_preds>=threshold)
	neg_hits = np.int32(neg_preds<threshold)
	acc = np.mean(np.concatenate([pos_hits, neg_hits], 1))
	return acc
