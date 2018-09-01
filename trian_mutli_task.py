from __future__ import print_function, absolute_import
import os
import sys
import time
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import data_manager
from dataset_loader import ImageDataset,AGE_Gender_ImageDataset
import transforms as T
import models
from losses import CrossEntropyLabelSmooth, DeepSupervision
from utils import AverageMeter, Logger, save_checkpoint
from optimizers import init_optim


parser = argparse.ArgumentParser(description='Age recognition:Train image model with cross entropy loss ')
# Architecture
parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=models.get_names())
parser.add_argument('-d', '--dataset', type=str, default='BOT',
					choices=data_manager.get_names())

parser.add_argument('--save-dir', type=str, default='log/831-mutil')
parser.add_argument('--cls_sample', type=int, default=150,
					help="each class  sample's temp(default: 256)")
parser.add_argument('--max-epoch', default=20, type=int,
					help="maximum epochs to run")
parser.add_argument('--train-batch', default=16, type=int,
					help="train batch size")
parser.add_argument('--test-batch', default=16, type=int, help="test batch size")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
					help="initial learning rate")
parser.add_argument('--height', type=int, default=160,
					help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=160,
					help="width of an image (default: 128)")
parser.add_argument('--gpu-devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

# Datasets
parser.add_argument('--root', type=str, default='data', help="root path to data directory")

parser.add_argument('-j', '--workers', default=4, type=int,
					help="number of data loading workers (default: 4)")

parser.add_argument('--split-id', type=int, default=0, help="split index")

# Optimization options
parser.add_argument('--optim', type=str, default='adam', help="optimization algorithm (see optimizers.py)")

parser.add_argument('--start-epoch', default=0, type=int,
					help="manual epoch number (useful on restarts)")

parser.add_argument('--stepsize', default=60, type=int,
					help="stepsize to decay learning rate (>0 means this is enabled)")
parser.add_argument('--gamma', default=0.1, type=float,
					help="learning rate decay")
parser.add_argument('--weight-decay', default=1e-6, type=float,
					help="weight decay (default: 5e-04)")

# Miscs
parser.add_argument('--print-freq', type=int, default=10, help="print frequency")
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH')
parser.add_argument('--evaluate', action='store_true', help="evaluation only")
parser.add_argument('--eval-step', type=int, default=1,
					help="run evaluation for every N epochs (set to -1 to test after training)")

parser.add_argument('--use-cpu', action='store_true', help="use cpu")


args = parser.parse_args()

def main():
	torch.manual_seed(args.seed)
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
	use_gpu = torch.cuda.is_available()
	if args.use_cpu: use_gpu = False

	if not args.evaluate:
		sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
	else:
		sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
	print("==========\nArgs:{}\n==========".format(args))

	if use_gpu:
		print("Currently using GPU {}".format(args.gpu_devices))
		cudnn.benchmark = True
		torch.cuda.manual_seed_all(args.seed)
	else:
		print("Currently using CPU (GPU is highly recommended)")

	print("Initializing dataset {}".format(args.dataset))

	dataset = data_manager.init_dataset(root=args.root, name=args.dataset,cls_sample = args.cls_sample)
	# print(dataset.train)
	# print(1)
	# 解释器：创建一个transform处理图像数据的设置
	# T.Random2DTranslation：随机裁剪
	# T.RandomHorizontalFlip: 给定概率进行随机水平翻转
	# T.ToTensor: 将PIL或numpy向量[0,255]=>tensor[0.0,1.0]
	# T.Normalize：用均值和标准偏差标准化张量图像，mean[ , , ]三个参数代表三通道
	transform_train = T.Compose([
		# T.RandomCrop(224),
		T.Random2DTranslation(args.height, args.width),
		T.RandomHorizontalFlip(),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])


	transform_test = T.Compose([
		# T.Resize(256),
		# T.CenterCrop(224),
		T.Resize((args.height, args.width)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

	pin_memory = True if use_gpu else False

	m = dataset.train
	print(1)
	# Dataloader 提供队列和线程
	# ImageDataset:return data =>img, pid, camid
	# RandomIdentitySampler:定义从数据集中抽取样本的策略
	# num_workers: 子进程数
	# print(dataset.train)
	trainloader = DataLoader(
		AGE_Gender_ImageDataset(dataset.train, transform=transform_train),
		batch_size=args.train_batch,shuffle=True, num_workers=args.workers,
		pin_memory=pin_memory, drop_last=True,
	)

	testloader = DataLoader(
		AGE_Gender_ImageDataset(dataset.test, transform=transform_test),
		batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
		pin_memory=pin_memory, drop_last=False,
	)

	print("Initializing model: {}".format(args.arch))
	model = models.init_model(name=args.arch, num_classes=dataset.train_num_class, loss={'xent'})
	print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

	# criterion_xent = CrossEntropyLabelSmooth(num_classes=dataset.train_num_class, use_gpu=use_gpu)
	age_criterion_xent = nn.CrossEntropyLoss()
	gender_criterion_xent = nn.CrossEntropyLoss()
	# optimizer = init_optim(args.optim, model.parameters(), args.lr, args.weight_decay)
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)
	if args.stepsize > 0:
		scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
	start_epoch = args.start_epoch

	if args.resume:
		print("Loading checkpoint from '{}'".format(args.resume))
		checkpoint = torch.load(args.resume)
		model.load_state_dict(checkpoint['state_dict'])
		start_epoch = checkpoint['epoch']

	if use_gpu:
		model = nn.DataParallel(model).cuda()

	if args.evaluate:
		print("Evaluate only")
		test(model, testloader, use_gpu)
		return

	start_time = time.time()
	train_time = 0
	# best_rank1 = -np.inf
	best_score = 0
	best_MAE = 0
	best_gender_acc =0
	best_epoch = 0
	print("==> Start training")

	for epoch in range(start_epoch, args.max_epoch):
		start_train_time = time.time()
		train(epoch, model, age_criterion_xent,gender_criterion_xent, optimizer, trainloader, use_gpu)
		train_time += round(time.time() - start_train_time)

		if args.stepsize > 0: scheduler.step()

		if args.eval_step > 0 and (epoch+1) % args.eval_step == 0 or (epoch+1) == args.max_epoch:
			print("==> Test")
			MAE ,Gender_acc = test(model, testloader, use_gpu)
			Score = Gender_acc*100 - MAE
			is_best = Score > best_score
			if is_best:
				best_score = Score
				best_MAE = MAE
				best_gender_acc = Gender_acc
				best_epoch = epoch + 1

			if use_gpu:
				state_dict = model.module.state_dict()
			else:
				state_dict = model.state_dict()
			save_checkpoint({
				'state_dict': state_dict,
				'rank1': Score,
				'epoch': epoch,
			}, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

	print("==> Best best_score(Gender_acc-MAE) {} |Gender_acc {}\t MAE {}|achieved at epoch {}".format(best_score,best_gender_acc,best_MAE,best_epoch))

	elapsed = round(time.time() - start_time)
	elapsed = str(datetime.timedelta(seconds=elapsed))
	train_time = str(datetime.timedelta(seconds=train_time))
	print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

def train(epoch, model, age_criterion_xent,gender_criterion_xent ,optimizer, trainloader, use_gpu):
	model.train()
	losses = AverageMeter()

	for batch_idx, (imgs, age_labels, gender_labels) in enumerate(trainloader):
		if use_gpu:
			imgs, age_labels, gender_labels = imgs.cuda(), age_labels.cuda(),gender_labels.cuda()
		age_outputs, gender_outputs = model(imgs)

		if isinstance(age_outputs, tuple):
			age_loss = DeepSupervision(age_criterion_xent, age_outputs, age_labels)
			gender_loss = DeepSupervision(gender_criterion_xent, gender_outputs, gender_labels)
		else:
			age_loss = age_criterion_xent(age_outputs, age_labels)
			gender_loss = gender_criterion_xent(gender_outputs, gender_labels)
		# loss = gender_loss
		loss = age_loss + gender_loss
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.update(loss.item(), age_labels.size(0))

		if (batch_idx+1) % args.print_freq == 0:
			print("Epoch {}/{}\t Batch {}/{}\t Loss {:.6f} ({:.6f})\t AgeLoss {:.6f}\t GenderLoss {:.6f}".format(
				epoch+1, args.max_epoch, batch_idx+1, len(trainloader), losses.val, losses.avg, age_loss, gender_loss
			))

def test(model, testloader, use_gpu):
	model.eval()
	gender_correct = 0
	total = 0
	MAE = 0
	with torch.no_grad():
		for batch_idx, (imgs, age_labels,gender_labels) in enumerate(testloader):
			if use_gpu:
				imgs ,age_labels, gender_labels = imgs.cuda(),age_labels.cuda(),gender_labels.cuda()
			age_outputs, gender_outputs = model(imgs)
			_, age_predicted = torch.max(age_outputs.data,1)
			_, gender_predicted = torch.max(gender_outputs.data,1)
			total += age_labels.size(0)

			# age_correct += (age_predicted == age_labels).sum()
			gender_correct += (gender_predicted == gender_labels).sum()
			MAE += torch.abs(age_predicted - age_labels).sum()

		MAE = MAE.cpu().numpy()
		gender_correct = gender_correct.cpu().numpy()

		MAE = float(MAE/total)
		gender_accurary = float(gender_correct/total)

		print('MAE:%2f'%MAE)
		print('gender acc:%.2f'%(gender_accurary*100))

	return MAE ,gender_accurary


if __name__ == '__main__':
	main()
