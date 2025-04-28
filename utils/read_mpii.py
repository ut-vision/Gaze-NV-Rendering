import csv
import numpy as np

def read_txt_as_dict(text_path):
	with open(text_path) as f:
		data = f.readlines()
	reader = csv.reader(data)
	p = {}
	for row in reader:
		words = row[0].split()
		p[words[0]] = words[1:]
	return p
def read_lm_gc(person_dict, index):
	landmarks = np.array([int(i) for i in person_dict[index][2:14]]).reshape((6,2))
	gc = np.array([float(i) for i in person_dict[index][23:26]]).reshape((3,1))
	return landmarks, gc
