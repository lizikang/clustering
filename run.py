import argparse
from data import Data
from cluster import Cluster


# define hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--vehicle_path', type=str, default='files/vehicle_info.txt')
parser.add_argument('--feature_path', type=str, default='files/features_10000.list')
parser.add_argument('--model_config_path', type=str, default='model/saver/config.json')

parser.add_argument('--topk', type=int, default=64)
parser.add_argument('--reid_threshold', type=float, default=0.7)
parser.add_argument('--max_cluster_size', type=int, default=15)
parser.add_argument('--faiss_search_type', type=str, default='dot product')
args = parser.parse_args()

# initialize data object
data = Data(args)

# initialize cluster object
cluster = Cluster(args)

# begin to cluster one by one
for i, track in enumerate(data.tracks):
	cluster.add(track)
	if (i+1) % 100 == 0: cluster.evaluate()
