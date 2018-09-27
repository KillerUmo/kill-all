import json
import cv2
from PIL import Image
import torch
import imutils
import numpy as np
import csv
import random
import multiprocessing as mp
import matplotlib.pyplot as plt
# # img = cv2.imread('./scene_1_00002.jpg')
# img_path ='./scene_1_00002.jpg'
# img = Image.open(img_path).convert('RGB')
# print(img.size)
# img.show()
# dateset = []
# with open('scene_1_00002.json','r') as load_f:
# 	load_dict = json.load(load_f)
#
# 	for person in load_dict['annotation'][0]['object']:
# 			print(person)
# 			position0 = (person['minx'],person['miny'])
# 			position1 = ((person['maxx'],person['maxy']))
# 			print(position0,position1)
# 			# cv2.rectangle(img,position0,position1,(0,0,0),1)
# 			dateset.append(('./scene_1_00002.jpg',position0,position1,
# 			                person['gender'],
# 			                person['staff'],
# 			                person['customer'],
# 			                person['stand'],
# 			                person['sit'],
# 			                person['play_with_phone']
# 			                ))
# 			# img1 = img[position0[1]:position1[1],position0[0]:position1[0]]
# 			print(9999999999)
# 			box  = (position0[0],position0[1],position1[0],position1[1])
# 			print(box)
# 			img1 = img.crop(box)
# 			img1.show()
# 	# img[c1[1]:c2[1], c1[0]:c2[0]]
# 	print(dateset)
# 	print(len(dateset))

def Confidence(fc_outputs, assign):
	a = fc_outputs[0][0].cpu().numpy()
	b = fc_outputs[0][1].cpu().numpy()
	print(a,b)
	if assign == 0:
		con = np.e**a/(np.e**a+np.e**b)
	else :
		con = np.e**b/(np.e**a+np.e**b)

	return '%.6f'%con

# gender_outputs = torch.Tensor([[-1.7,1.8]])
# conf_male = Confidence(gender_outputs,0)
# print(conf_male)
# staff_con = 2
# customer_con = 2
# sf = 'staff' if staff_con >= customer_con else 'customer'
# print(sf)

# print('请输入当前行人属性(0/1)：男/女|站/坐|店员/顾客|是否玩手机#间隙用空格隔开')
# get = input()
# print(get)
# print(type(get))
# gender = get.split(' ')[1]
# print(gender)

# ran = round(random.uniform(0.6,1),6)
# ran1 = random.uniform(0.6,1)
# ran2= random.uniform(0.6,1)
# ran3 = random.uniform(0.6,1)
# print(ran,ran1,ran2,ran3)
# print(type(ran))
# p = round(1-ran,6)
# print(p)
# load_dict = {'tets':1}
# with open('./result/test_lh.json', 'r') as load_f:
# with open('./test95.json', 'r') as load_f:
# 	load_dict = json.load(load_f)
# del load_dict['results'][0]['object'][0]
# with open("./test95-2.json", "w") as dump_f:
# 	json.dump(load_dict, dump_f)

def img_show(q):
	print('hahahha')
	# if q.get():
	# 	print('get')
	# 	img = q.get()
	# 	cv2.imshow('person',img)
	# 	cv2.waitKey(10)



# # mp.set_start_method('spawn')
# queue = mp.Queue(maxsize=2)
# p = mp.Process(target=img_show, args=(queue,))
# p.daemon = True  # 设置为daemon的线程会随着主线程的退出而结束，而非daemon线程会阻塞主线程的退出。
# p.start()
#
# img = cv2.imread('./1.jpg')
# queue.put(img)
# # cv2.imshow('img',img)
#
# while(1):
# 	print(1)
# 	# cv2.waitKey(5)
# 	name = input()
def Image_rectangle(img,c1,c2,name):
	# c1 = tuple(c1.int())
	# c2 = tuple(c2.int())
	color = (0, 0, 255)
	cv2.rectangle(img, c1, c2, color,1)    # 加框
	t_size = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
	c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
	cv2.rectangle(img, c1, c2, color, -1) # -1填充作为文字框底色
	cv2.putText(img, str(name), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
	return img

if __name__ == '__main__':
	img = cv2.imread('./data/scene_5_00009.jpg')
	with open('./result/id_7778_hand.json','r') as load:
		load_dict = json.load(load)

	for i,person in enumerate(load_dict['results'][0]['object']):
		img = Image_rectangle(img, (person['minx'],person['miny']),  (person['maxx'],person['maxy']), str(i))
	cv2.imshow('img',img)
	cv2.waitKey(0)
