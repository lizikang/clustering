import argparse
from data import Data
from model import Model

import os
cur_dir = os.path.abspath(os.path.dirname(__file__))
up_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


# define hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=os.path.join(up_dir, 'files/vehicle_info.txt'))
parser.add_argument('--model_path', type=str, default=os.path.join(cur_dir, 'saver'))

parser.add_argument('--seq_len', type=int, default=4)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--sample_ratio', type=int, default=1)
parser.add_argument('--hidden_units', type=int, default=256)

parser.add_argument('--num_epochs', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--learn_rate', type=float, default=3e-4)
parser.add_argument('--dropout_prob', type=float, default=0.5)
args = vars(parser.parse_args())

# initialize data object
data = Data(args)

# initialize model object
args['camera_size'] = data.camera_size 
model = Model(args)

# train and inference
model.train(data.train_data, data.valid_data)
model.inference(data.test_data)
