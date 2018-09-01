from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave

from utils import mkdir_if_missing, write_json, read_json

# FG-NET读取

class FG(object):
	dataset_dir = 'fg-net'

	def __init__(self, root='data', **kwargs):
		self.dataset_dir = osp.join(root, self.dataset_dir)

		self.train_dir = self.dataset_dir
		# self.test_dir = self.dataset_dir
		self.test_dir = osp.join(osp.join(root, 'LAP2'), 'test_clean_70')
		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> FG loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | ".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

		# self.num_train_pids = num_train_pids
		# self.num_query_pids = num_query_pids
		# self.num_gallery_pids = num_gallery_pids

	def get_files(self,file_dir):
		dateset = []
		for label in os.listdir(file_dir):
			subfile_dir = os.path.join(file_dir, label)
			for picfile in os.listdir(subfile_dir):
				picdir = os.path.join(subfile_dir, picfile)
				dateset.append((picdir, int(label)))
		num_class = len(os.listdir(file_dir))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		# if not osp.exists(self.query_dir):
		#     raise RuntimeError("'{}' is not available".format(self.query_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))


class Imdb_Wiki(object):
	dataset_dir = 'Imdb_Wiki'

	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train_clean')
		self.test_dir = osp.join(self.dataset_dir, 'test_clean')
		# self.train_dir = self.dataset_dir
		# self.test_dir = self.dataset_dir
		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> Imdb_Wiki loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | ".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for label in os.listdir(file_dir):
			m = 0
			subfile_dir = os.path.join(file_dir, label)
			for picfile in os.listdir(subfile_dir):
				picdir = os.path.join(subfile_dir, picfile)
				dateset.append((picdir, int(label)))
				m +=1
				if m == self.cls_sample:
					break
		num_class = len(os.listdir(file_dir))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))

class Imdb_Wiki_70(object):
	dataset_dir = 'Imdb_Wiki'

	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train_clean_70')
		# self.test_dir = osp.join(self.dataset_dir, 'test_clean_70')
		self.test_dir = osp.join(osp.join(root, 'LAP2'), 'test_clean_70')
		# self.train_dir = self.dataset_dir
		# self.test_dir = self.dataset_dir
		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> Imdb_Wiki_70 loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | ".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for label in os.listdir(file_dir):
			m = 0
			subfile_dir = os.path.join(file_dir, label)
			for picfile in os.listdir(subfile_dir):
				picdir = os.path.join(subfile_dir, picfile)
				dateset.append((picdir, int(label)))
				m +=1
				if m == self.cls_sample:
					break
		num_class = len(os.listdir(file_dir))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))


class LAP(object):
	dataset_dir = 'LAP'
	def __init__(self, root='data',cls_sample = 500, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train_clean')
		self.test_dir = osp.join(self.dataset_dir, 'test_clean')
		# self.train_dir = self.dataset_dir
		# self.test_dir = self.dataset_dir
		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> LAP loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | ".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for label in os.listdir(file_dir):
			m = 0
			subfile_dir = os.path.join(file_dir, label)
			for picfile in os.listdir(subfile_dir):
				picdir = os.path.join(subfile_dir, picfile)
				dateset.append((picdir, int(label)))
				m +=1
				if m == self.cls_sample:
					break
		num_class = len(os.listdir(file_dir))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))

class LAP_70(object):
	dataset_dir = 'LAP'
	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train_clean_70')
		self.test_dir = osp.join(self.dataset_dir, 'test_clean_70')
		# self.train_dir = self.dataset_dir
		# self.test_dir = self.dataset_dir
		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> LAP_70 loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | ".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for label in os.listdir(file_dir):
			m = 0
			subfile_dir = os.path.join(file_dir, label)
			for picfile in os.listdir(subfile_dir):
				picdir = os.path.join(subfile_dir, picfile)
				dateset.append((picdir, int(label)))
				m +=1
				if m == self.cls_sample:
					break
		num_class = len(os.listdir(file_dir))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))

class LAP2_70(object):
	dataset_dir = 'LAP2'
	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train_clean_70')
		self.test_dir = osp.join(self.dataset_dir, 'test_clean_70')
		# self.train_dir = self.dataset_dir
		# self.test_dir = self.dataset_dir
		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> LAP2_70 loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | ".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for label in os.listdir(file_dir):
			m = 0
			subfile_dir = os.path.join(file_dir, label)
			for picfile in os.listdir(subfile_dir):
				picdir = os.path.join(subfile_dir, picfile)
				dateset.append((picdir, int(label)))
				m +=1
				if m == self.cls_sample:
					break
		num_class = len(os.listdir(file_dir))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))

class LAP2_large_70(object):
	dataset_dir = 'LAP2'
	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train_large_70')
		self.test_dir = osp.join(self.dataset_dir, 'test_large_70')
		# self.train_dir = self.dataset_dir
		# self.test_dir = self.dataset_dir
		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> LAP2_large_70 loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | ".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for label in os.listdir(file_dir):
			m = 0
			subfile_dir = os.path.join(file_dir, label)
			for picfile in os.listdir(subfile_dir):
				picdir = os.path.join(subfile_dir, picfile)
				dateset.append((picdir, int(label)))
				m +=1
				if m == self.cls_sample:
					break
		num_class = len(os.listdir(file_dir))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))

'''
性别识别数据集
Imdb:训练集
Wiki:测试集
'''
class Imdb_Wiki_Gender(object):
	dataset_dir = 'Gender'
	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train_imdb')
		self.test_dir = osp.join(self.dataset_dir, 'test_wiki')
		# self.train_dir = self.dataset_dir
		# self.test_dir = self.dataset_dir
		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> Imdb_Wiki_Gender loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | ".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for label in os.listdir(file_dir):
			subfile_dir = os.path.join(file_dir, label)
			for picfile in os.listdir(subfile_dir):
				picdir = os.path.join(subfile_dir, picfile)
				dateset.append((picdir, int(label)))

		num_class = len(os.listdir(file_dir))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))

'''
性别识别数据集
Imdb:训练集
Wiki:测试集
相比于Imdb_Wiki_Gender，Imdb_Wiki_Gender2的图片包含更多的人脸周围信息（1.35）
'''
class Imdb_Wiki_Gender2(object):
	dataset_dir = 'Gender2'
	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train_imdb')
		self.test_dir = osp.join(self.dataset_dir, 'test_wiki')
		# self.train_dir = self.dataset_dir
		# self.test_dir = self.dataset_dir
		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> Imdb_Wiki_Gender loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | ".format( len(train)))
		print("  test  | {:5d} | ".format(len(test)))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for label in os.listdir(file_dir):
			subfile_dir = os.path.join(file_dir, label)
			for picfile in os.listdir(subfile_dir):
				picdir = os.path.join(subfile_dir, picfile)
				dateset.append((picdir, int(label)))

		num_class = len(os.listdir(file_dir))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))

'''
多任务
年龄-性别识别数据集
Imdb:训练集
Wiki:测试集

'''
class Age_Gender(object):
	dataset_dir = 'Age_Gender'
	def __init__(self, root='data',cls_sample = 30, **kwargs):
		self.cls_sample = cls_sample
		self.dataset_dir = osp.join(root, self.dataset_dir)
		self.train_dir = osp.join(self.dataset_dir, 'train_imdb')
		# self.train_dir = osp.join(self.dataset_dir, 'test_wiki')
		self.test_dir = osp.join(self.dataset_dir, 'test_wiki')

		self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> Age_Gender loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | num_class: {}|".format( len(train),train_num_class))
		print("  test  | {:5d} | num_class: {}|".format(len(test),train_num_class))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		for gender_label in os.listdir(file_dir):
			gender_subfile_dir = os.path.join(file_dir, gender_label)
			for age_label in os.listdir(gender_subfile_dir):
				age_subfile_dir = os.path.join(gender_subfile_dir,age_label)
				for picfile in os.listdir(age_subfile_dir):
					picdir = os.path.join(age_subfile_dir, picfile)
					dateset.append((picdir,int(age_label), int(gender_label)))

		num_class = len(os.listdir(os.path.join(file_dir, os.listdir(file_dir)[0])))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))


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

		# self._check_before_run()

		train, train_num_class = self.get_files(self.train_dir)
		test, test_num_class = self.get_files(self.test_dir)

		print("=> BOT loaded")
		print("Dataset statistics:")
		print("  ------------------------------")
		print("  train    | {:5d} | num_class: {}|".format( len(train),train_num_class))
		print("  test  | {:5d} | num_class: {}|".format(len(test),train_num_class))
		print("  ------------------------------")

		self.train = train
		self.test = test
		self.train_num_class = train_num_class

	def get_files(self,file_dir):
		dateset = []
		images_dir = os.path.join(file_dir,'image')
		labels_dir = os.path.join(file_dir,'label')

		for image_dir in os.listdir(images_dir):
			print(image_dir)
			label_dir = os.path.join(labels_dir,image_dir.split(".")[0])
			image_dir = os.path.join(images_dir,image_dir)
			dateset.append((image_dir))
			print(label_dir,image_dir)
		num_class = len(os.listdir(images_dir))
		# for gender_label in os.listdir(file_dir):
		# 	gender_subfile_dir = os.path.join(file_dir, gender_label)
		# 	for age_label in os.listdir(gender_subfile_dir):
		# 		age_subfile_dir = os.path.join(gender_subfile_dir,age_label)
		# 		for picfile in os.listdir(age_subfile_dir):
		# 			picdir = os.path.join(age_subfile_dir, picfile)
		# 			dateset.append((picdir,int(age_label), int(gender_label)))
		#
		# num_class = len(os.listdir(os.path.join(file_dir, os.listdir(file_dir)[0])))
		return dateset, num_class

	def _check_before_run(self):
		"""Check if all files are available before going deeper"""
		if not osp.exists(self.dataset_dir):
			raise RuntimeError("'{}' is not available".format(self.dataset_dir))
		if not osp.exists(self.train_dir):
			raise RuntimeError("'{}' is not available".format(self.train_dir))
		if not osp.exists(self.test_dir):
			raise RuntimeError("'{}' is not available".format(self.test_dir))
"""Create dataset"""

__factory = {
	# image-based
	'fg-net': FG,
	'Imdb_Wiki':Imdb_Wiki,
	'Imdb_Wiki_70': Imdb_Wiki_70,
	'LAP': LAP,
	'LAP_70': LAP_70,
	'LAP2_70': LAP2_70,
	'LAP2_large_70': LAP2_large_70,
	'Imdb_Wiki_Gender' : Imdb_Wiki_Gender,
	'Imdb_Wiki_Gender2': Imdb_Wiki_Gender2,
	'Age_Gender': Age_Gender,
	'BOT': BOT,
}



def get_names():
	return __factory.keys()

def init_dataset(name, **kwargs):
	if name not in __factory.keys():
		raise KeyError("Unknown dataset: {}".format(name))
	return __factory[name](**kwargs)

if __name__ == '__main__':
	pass