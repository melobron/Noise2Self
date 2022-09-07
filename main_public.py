import argparse
from train_public import TrainN2S

# Arguments
parser = argparse.ArgumentParser(description='Train N2S public')

parser.add_argument('--exp_detail', default='Train N2S public', type=str)
parser.add_argument('--gpu_num', default=0, type=int)
parser.add_argument('--seed', default=100, type=int)

# Training parameters
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--start_epoch', default=0, type=int)
parser.add_argument('--decay_epoch', default=50, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--noise', default='gauss_25', type=str)
parser.add_argument('--masker_width', default=4, type=int)
parser.add_argument('--masker_mode', default='interpolate', type=str)

# Transformations
parser.add_argument('--crop', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--mean', type=float, default=0.4050)  # ImageNet Gray: 0.4050
parser.add_argument('--std', type=float, default=0.2927)  # ImageNet Gray: 0.2927

args = parser.parse_args()

# Train N2S
train_N2S = TrainN2S(args=args)
train_N2S.train()
