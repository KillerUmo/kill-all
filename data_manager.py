from __future__ import print_function, absolute_import
import os
import json
import os.path as osp
# import glob
# import re
# import sys
# import urllib
# import tarfile
# import zipfile

# from scipy.io import loadmat
# import numpy as np
# import h5py
# from scipy.misc import imsave
#
# from utils import mkdir_if_missing, write_json, read_json

'''
多任务
BOT行人属性数据集
train:训练集
test:测试集

'''
class BOT(object):
	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		# self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(root, 'train')
		self.test_dir = osp.join(root, 'test')

		train = self.get_files(self.train_dir)
		test = self.get_files(self.test_dir)

		print("=> BOT loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} |".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		# self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		images_dir = os.path.join(file_dir,'image')
		labels_dir = os.path.join(file_dir,'label')

		for image_dir in os.listdir(images_dir):
			label_dir = os.path.join(labels_dir,image_dir.split(".")[0] + ".json")
			image_dir = os.path.join(images_dir,image_dir)

			with open(label_dir, 'r') as load_f:
				load_dict = json.load(load_f)
				for person in load_dict['annotation'][0]['object']:
					position0 = (person['minx'], person['miny'])
					position1 = ((person['maxx'], person['maxy']))

					dateset.append((image_dir,
					                position0, position1,
					                person['gender'],
					                person['staff'],
					                person['customer'],
					                person['stand'],
					                person['sit'],
					                person['play_with_phone']
					                ))
		return dateset

'''
多任务
BOT行人属性数据集
train:训练集
test:测试集

'''
class wuma_bot(object):
	dataset_dir = 'wuma'
	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'crop_train')
		self.test_dir = osp.join(self.dataset_dir, 'crop_test')

		train = self.get_files(self.train_dir)
		test = self.get_files(self.test_dir)

		print("=> WuMa BOT loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} |".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		# self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for image_dir in os.listdir(file_dir):
			gender = int(image_dir.split('_')[4])
			staff = int(image_dir.split('_')[5])
			customer = int(image_dir.split('_')[6])
			stand = int(image_dir.split('_')[7])
			sit = int(image_dir.split('_')[8])
			play_with_phone = int(image_dir.split('_')[9].split('.')[0])

			image_dir = os.path.join(file_dir,image_dir)

			dateset.append((image_dir,gender,staff,customer,stand,sit,play_with_phone))
		return dateset


"""Create dataset"""

__factory = {
	'BOT': BOT,
	'wuma_bot':wuma_bot
}



def get_names():
	return __factory.keys()

def init_dataset(name, **kwargs):
	if name not in __factory.keys():
		raise KeyError("Unknown dataset: {}".format(name))
	return __factory[name](**kwargs)

if __name__ == '__main__':
	pass