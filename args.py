
import argparse


arg_lists = []
parser = argparse.ArgumentParser(description='RAM')


def add_argument_group(name):
	arg = parser.add_argument_group(name)
	arg_lists.append(arg)
	return arg

def str2bool(v):
	return v.lower() in ('true', '1')
	

parser = argparse.ArgumentParser()
## xgaze
parser.add_argument('-save', '--save_dir', type=str)
parser.add_argument('-real', '--real_dir', type=str, default=None)
parser.add_argument('-ablation', '--ablation', type=bool, default=False)
parser.add_argument('--group', type=int, help='there are 4 groups of subjects, which group to use (just for parallel rendering)')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

