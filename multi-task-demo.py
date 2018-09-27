import cv2
import torch
from PIL import Image
import json
from process import *
import models
import os
import glob
import argparse
import csv


parser = argparse.ArgumentParser(description='Age recognition:video and image ')
parser.add_argument('-a', '--arch_BOT', type=str, default='ResNet50_BOT_MultiTask', choices=models.get_names())
parser.add_argument('--resume_BOT', type=str, default='log/92-multi/best_model.pth.tar', metavar='PATH')
# parser.add_argument('--resume_BOT', type=str, default='log/910-resnet50-wuma/best_model.pth.tar', metavar='PATH')
parser.add_argument('--gpu-devices', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

'''
1.读取检测框的json文件
2.写入多任务BOT模型的结果
3.生成最终提交的文件
'''
def get_result(val1_dir,json_in,json_out):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
	model_BOT = Init_BOT_model(args.resume_BOT,args.arch_BOT)
	imgs_dir = get_imgs_dir(val1_dir)

	with open(json_in,'r') as load_f:
		load_dict = json.load(load_f)

		for i, image_result in enumerate(load_dict['results']):
			print('第%d张图片'%i)
			persons = image_result['object']
			img = Image.open(imgs_dir[i]).convert('RGB')
			for m, person in enumerate(persons):
				box = (person['minx'],person['miny'],person['maxx'],person['maxy'])
				img_crop = img.crop(box)
				# img_crop.show()
				staff_con, customer_con, stand_con, sit_con, play_with_phone_con, male_con, female_con = BOT_recognition(model_BOT,img_crop)
				load_dict['results'][i]['object'][m]['staff'] = staff_con
				load_dict['results'][i]['object'][m]['customer'] = customer_con
				load_dict['results'][i]['object'][m]['stand'] = stand_con
				load_dict['results'][i]['object'][m]['sit'] = sit_con
				load_dict['results'][i]['object'][m]['play_with_phone'] = play_with_phone_con
				load_dict['results'][i]['object'][m]['male'] = male_con
				load_dict['results'][i]['object'][m]['female'] = female_con
				load_dict['results'][i]['object'][m]['confidence'] = 1.0

	with open(json_out,"w") as dump_f:
		json.dump(load_dict, dump_f)
		print('save achieve')

'''
功能：识别训练集属性
'''
def get_train_result(train_dir):
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
	model_BOT = Init_BOT_model(args.resume_BOT,args.arch_BOT)
	imgs_dir = []
	for each_img in sorted(os.listdir(train_dir)):
		each_img = os.path.join(train_dir, each_img)
		imgs_dir.append(each_img)

	with open('./result/test_train.json','r') as load_f:
		load_dict = json.load(load_f)

		for i, image_result in enumerate(load_dict['results']):
			print('第%d张图片'%i)

			persons = image_result['object']
			img = Image.open(imgs_dir[i]).convert('RGB')

			for m, person in enumerate(persons):
				box = (person['minx'],person['miny'],person['maxx'],person['maxy'])
				img_crop = img.crop(box)

				staff_con, customer_con, stand_con, sit_con, play_with_phone_con, male_con, female_con = BOT_recognition(model_BOT,img_crop)
				load_dict['results'][i]['object'][m]['staff'] = staff_con
				load_dict['results'][i]['object'][m]['customer'] = customer_con
				load_dict['results'][i]['object'][m]['stand'] = stand_con
				load_dict['results'][i]['object'][m]['sit'] = sit_con
				load_dict['results'][i]['object'][m]['play_with_phone'] = play_with_phone_con
				load_dict['results'][i]['object'][m]['male'] = male_con
				load_dict['results'][i]['object'][m]['female'] = female_con
				load_dict['results'][i]['object'][m]['confidence'] = 1.0

	with open("./result/group1_train_20180907_1.json","w") as dump_f:
		json.dump(load_dict, dump_f)
		print('save achieve')

'''
功能：得到验证集图片的具体路径列表

输入：验证集地址
输出：验证集中所有图片的地址
'''
def get_imgs_dir(val1_dir):
	imgs_dir = []
	for each_scene in sorted(os.listdir(val1_dir)):
		each_scene = os.path.join(val1_dir,each_scene)
		for each_img in sorted(os.listdir(each_scene)):
			each_img = os.path.join(each_scene, each_img)
			imgs_dir.append(each_img)
	return imgs_dir

'''
查看json的内容
'''
def look(json_dir):
	with open(json_dir,'r') as load_f:
		load_dict = json.load(load_f)
		print(type(load_dict))
		print(type(load_dict['results']),len(load_dict['results']))
		print(load_dict['results'][0])



'''
给图片加框和名字
'''
def Image_rectangle(img,c1,c2,name):
	color = (0, 0, 255)
	cv2.rectangle(img, c1, c2, color,1)    # 加框
	t_size = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
	cv2.putText(img, str(name), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
	return img


'''
功能：从json的个人标签中整合成新的简洁的标注信息
如:(male:1,female:0)转化为gender:'male'

'''
def get_message(person):
	staff_con = float(person['staff'])
	customer_con = float(person['customer'])
	stand_con = float(person['stand'])
	sit_con = float(person['sit'])
	play_with_phone_con = float(person['play_with_phone'])
	male_con = float(person['male'])
	female_con = float(person['female'])

	id = 'staff' if staff_con >= customer_con else 'customer'
	pos = 'stand' if stand_con >= sit_con else 'sit'
	phone = 'phone' if play_with_phone_con >= 0.5 else 'nophone'
	gender = 'male' if male_con >= female_con else 'female'

	message = str(id) + str('-') + str(pos) + str('-') + str(phone) + str('-') + str(gender)

	return message


'''
可视化val1-json的结果
输入：
	1） val1_dir：验证集1的地址
	2） output_dir：输出加标注验证集1的地址
	3） json_dir：json的地址
输出：加框加属性识别信息的图片集和，一般放在result文件夹下
'''
def view_result(val1_dir,output_dir,json_dir):
	print('start')
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	imgs_dir = get_imgs_dir(val1_dir)
	with open(json_dir, 'r') as load_f:
		load_dict = json.load(load_f)

		for i, image_result in enumerate(load_dict['results']):
			persons = image_result['object']
			img = cv2.imread(imgs_dir[i])
			img_outdir = os.path.join(output_dir ,os.path.basename(imgs_dir[i]))
			for m, person in enumerate(persons):
				message = get_message(person)
				img = Image_rectangle(img,(person['minx'],person['miny']),(person['maxx'],person['maxy']),message)
			cv2.imwrite(img_outdir,img)
	print('achieve')


'''
可视化train-json的结果
输入：
	1） train_dir：训练集的地址
	2） output_dir：输出加标注验证集1的地址
	3） json_dir：json的地址
'''
def view_train_result(train_dir,output_dir,json_dir):
	print('start')
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	# imgs_dir = get_imgs_dir(val1_dir)
	imgs_dir = []
	for each_img in sorted(os.listdir(train_dir)):
		each_img = os.path.join(train_dir, each_img)
		imgs_dir.append(each_img)

	with open(json_dir, 'r') as load_f:
		load_dict = json.load(load_f)

		for i, image_result in enumerate(load_dict['results']):
			print('第%d张图片'%i)
			persons = image_result['object']
			img = cv2.imread(imgs_dir[i])
			img_outdir = os.path.join(output_dir ,os.path.basename(imgs_dir[i]))
			for m, person in enumerate(persons):
				message = get_message(person)
				img = Image_rectangle(img,(person['minx'],person['miny']),(person['maxx'],person['maxy']),message)
			cv2.imwrite(img_outdir,img)

'''
功能：保存单个场景的json
输入：1）json_in：完整场景的json地址
	  2）json_out：单个场景输出的json地址
	  3）scene_num：需要保留的场景（如场景一，则输入1）
'''
def keep_scene(json_in,json_out,scene_num):
	with open(json_in,'r') as load_f:
		load_dict = json.load(load_f)
	csv_reader = csv.reader(open("./data/val1_filename_id.csv"))
	file2id = {}
	final_json = {'results': [] }
	for row in csv_reader:
		file2id[row[1]] = row[0]

	for m,image in enumerate(load_dict['results']):
		scene_read  = int(file2id[image['image_id']].split('_')[1])
		if scene_read == scene_num:
			final_json['results'].append(load_dict['results'][m])
			print('场景%d,记录'%scene_num)
		else :
			print('--跳过场景%d'%scene_read)

	with open(json_out,"w") as dump_f:
		json.dump(final_json, dump_f)
		print('save achieve')

if __name__ == '__main__':
	# view_result('./data/val1', './result/val1_result_920_1', './result/group1_val1_20180920_1.json')
	# view_train_result('./data/train/image', './result/train_result_98', './result/group1_train_20180907_1.json')
	# look('./result/scene1.json')
	get_result('./data/val1',json_in='./result/test_jw927_1.json',json_out="./result/group1_val1_20180927_1.json")
	# keep_scene(json_in="./result/group1_val1_20180923_3.json",json_out='result/923_scene2.json',scene_num=2)
	# keep_scene(json_in="./result/group1_val1_20180923_3.json",json_out='result/923_scene3.json',scene_num=3)
	# keep_scene(json_in="./result/group1_val1_20180920_1.json",json_out='result/921_scene5.json',scene_num=5)