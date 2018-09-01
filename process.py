import face_recognition
import models
import torch
import torch.nn as nn
import transforms as T
from dataset_loader import Load_person
from torch.utils.data import DataLoader
from PIL import Image
import cv2

'''
输入：
	1）age_dir:模型存放地址（pytorch）
	2）age_cfg:base模型类别，如resnet18,resnet50等
输出：
	age:预测的年龄
'''
def Init_age_model(age_dir,age_cfg):
	CUDA = torch.cuda.is_available()
	print("Initializing model: {}".format(age_cfg))
	model_age = models.init_model(name=age_cfg, num_classes=70, loss={'xent'}, use_gpu=CUDA)
	if age_dir:
		print("Loading checkpoint from '{}'".format(age_dir))
		checkpoint = torch.load(age_dir)
		model_age.load_state_dict(checkpoint['state_dict'])
		print("Age Network successfully loaded")
	if CUDA:
		model_age = nn.DataParallel(model_age).cuda()
	return model_age


'''
输入：
	1）gender_dir:模型存放地址（pytorch）
	2）gender_cfg:base模型类别，如resnet18,resnet50等
输出：
	age:预测的年龄
'''
def Init_gender_model(gender_dir,gender_cfg):
	CUDA = torch.cuda.is_available()
	print("Initializing model: {}".format(gender_cfg))
	model_age = models.init_model(name=gender_cfg, num_classes=2, loss={'xent'}, use_gpu=CUDA)
	if gender_dir:
		print("Loading checkpoint from '{}'".format(gender_dir))
		checkpoint = torch.load(gender_dir)
		model_age.load_state_dict(checkpoint['state_dict'])
		print("Age Network successfully loaded")
	if CUDA:
		model_age = nn.DataParallel(model_age).cuda()
	return model_age

'''
输入：
	1）model_age:年龄识别模型（pytorch）
	2）img:人脸裁剪图片
输出：
	age:预测的年龄
'''
def age_recognition(model_age,img):
	img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	transform = T.Compose([
		T.Resize((256, 128)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	loader = DataLoader(
		Load_person(img, transform=transform),
		batch_size=1, shuffle=False, num_workers=0,
		pin_memory=True, drop_last=False, )
	model_age.eval()
	with torch.no_grad():
		for batch_idx,img2 in enumerate(loader):
			if torch.cuda.is_available():img2 = img2.cuda()
			score = model_age(img2)
			age = torch.argmax(score.data,1)
			age = age[0].cpu().numpy()

	return age

'''
输入：
	1）model_gender:性别识别模型（pytorch）
	2）img:人脸裁剪图片（包括头发）
输出：
	gender:预测的性别
'''
def gender_recognition(model_gender,img):
	img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
	transform = T.Compose([
		T.Resize((256, 128)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	loader = DataLoader(
		Load_person(img, transform=transform),
		batch_size=1, shuffle=False, num_workers=0,
		pin_memory=True, drop_last=False, )
	model_gender.eval()
	with torch.no_grad():
		for batch_idx,img2 in enumerate(loader):
			if torch.cuda.is_available():img2 = img2.cuda()
			score = model_gender(img2)
			gender = torch.argmax(score.data,1)
			gender = gender[0].cpu().numpy()

	return gender

'''
人脸检测
输入：单张图片
输出：裁剪后的人脸，是否成功检测标志位
备注：cnn模式： 服务器上0.01s一张 ，PC上8s一张
	 无cnn模式：服务器上0.09s一张，PC上0.3s一张
'''
def face_det(img):
	# face_locations = face_recognition.face_locations(img, model="cnn")
	face_locations = face_recognition.face_locations(img)
	if len(face_locations):
		(y0, x1, y1, x0) = face_locations[0]
		imgcopy = img.copy()
		img_cut = imgcopy[y0:y1,x0:x1]
		return img_cut,True,face_locations
	else:
		return img ,False,None


'''
输入：
	1）image:摄像头得到的未裁剪图片
	2）face_locations:人脸位置坐标
	3) info:预测的年龄、性别等信息，是一个字符串
	4) 
输出：
	img:加框加年龄备注之后的画面
'''
def Information_show(img,face_locations,info,single=False):
	if not single:
		for (y0, x1, y1, x0) in face_locations:
			cv2.rectangle(img, (x0, y0), (x1, y1), ( 0, 0,255), 2)
			info = str(info)
			t_size = cv2.getTextSize(str(info), cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
			x2,y2 = x0 + t_size[0] + 3, y0 + t_size[1] + 4
			cv2.rectangle(img, (x0,y0), (x2,y2), (0, 0, 255), -1)  # -1填充作为文字框底色
			cv2.putText(img, info, (x0, y0 +t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

	else:
		(y0, x1, y1, x0) = face_locations
		cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
		info = str(info)
		t_size = cv2.getTextSize(str(info), cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
		x2, y2 = x0 + t_size[0] + 3, y0 + t_size[1] + 4
		cv2.rectangle(img, (x0, y0), (x2, y2), (0, 0, 255), -1)  # -1填充作为文字框底色
		cv2.putText(img, info, (x0, y0 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

	return img


def Recognition_age_gender(img,face_location,model_age,model_gender):
	(y0, x1, y1, x0) = face_location

	# x0_ = int(1.2 * x0 - 0.2 * x1)
	# y0_ = int(1.2 * y0 - 0.2 * y1)
	# x1_ = int(1.2 * x1 - 0.2 * x0)
	# y1_ = int(1.2 * y1 - 0.2 * y0)
	x0_ = int(1.3 * x0 - 0.3 * x1)
	y0_ = int(1.3 * y0 - 0.3 * y1)
	x1_ = int(1.3 * x1 - 0.3 * x0)
	y1_ = int(1.3 * y1 - 0.3 * y0)

	x0_ = x0_ if x0_ > 0 else 0
	y0_ = y0_ if y0_ > 0 else 0
	x1_ = x1_ if x1_ < img.shape[1] else img.shape[1]
	y1_ = y1_ if y1_ < img.shape[0] else img.shape[0]

	age_face = img[y0:y1,x0:x1]
	gender_face = img[y0_:y1_,x0_:x1_]
	# cv2.imshow('age_face',age_face)
	# cv2.imshow('gender_face',gender_face)
	age = age_recognition(model_age, age_face)
	gender = gender_recognition(model_gender,gender_face)
	print('age',age)
	print('gender',gender)

	return age,gender