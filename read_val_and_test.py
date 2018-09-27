import cv2
import json
import glob
import csv
from camera import *
import os

'''
将验证集图片导入模型进行框的提取，并按照要求输出json文件
'''
def lh():
	final_json = {'results': [] }


	model_yolo3 = Init_yolo("yolov3.weights", "cfg/yolov3.cfg")
	num = 0

	data_test = "../dataset/test/"
	for item in glob.glob(data_test + '*'):
		for image in glob.glob(item + '/*'):
			print(image)
			# 这部分是为了把id与图片的名称对应起来
			image_id = (image.split('\\')[2]).split('_')
			if int(image_id[1]) == 1:
				final_id = int(image_id[2].split('.')[0])
			elif int(image_id[1]) == 2:
				final_id = int(image_id[2].split('.')[0]) + 3520
			elif int(image_id[1]) == 3:
				final_id = int(image_id[2].split('.')[0]) + 3520 + 1745
			elif int(image_id[1]) == 4:
				final_id = int(image_id[2].split('.')[0]) + 3520 + 1745 + 1848
			elif int(image_id[1]) == 5:
				final_id = int(image_id[2].split('.')[0]) + 3520 + 1745 + 1848 + 656
			print(final_id)
			final_id = "id_" + str(final_id)
			final_json["results"].append({"image_id": final_id, "object":[]})
			image = cv2.imread(image)

			# 下面是模型处理
			outputs, haveperson = yolo3(model_yolo3, image, biggest_per=False)
			average = 0
			if haveperson:
				for i, output in enumerate(outputs):
					print(i, output)
					# img类型是一个narry,所以用索引的话格式为img[:,:,:]
					print(int((output[3] - output[1]) * 0.2 + output[1]))  # 高
					print(((output[4] - output[2]) * 0.2 + output[2]))
					# print(img[0:300, 0:300, 0:3].shape)
					count = image[int(output[2]):int((output[4] - output[2]) * 0.4 + output[2]),
							int(output[1]):int((output[3] - output[1]) * 1 + output[1]), 0:3]
					# 这里是表示shape输出，先为高，再为长，最后是深度
					gao, chang, shen = count.shape
					print(gao, chang, shen)
					for l in range(gao):
						for j in range(chang):
							for k in range(3):
								average += count[l, j, k]
					average = average / (l * j * k)
					# 当均值小于180的时候不计入
					if average < 180:
						# 每次图片识别完之后加入框
						final_json["results"][num]["object"].append({"minx": int(output[1]),
																  "miny": int(output[2]),
																  "maxx": int(output[3]),
																  "maxy": int(output[4]),
																  "staff": -1, "customer": -1, "stand": -1,  "sit": -1,
																  "play_with_phone": -1, "male": -1,
																  "female": -1, "confidence": -1})

					else:
						i -= 1


			else:
				print("Yolo3 can not detect person")
			num += 1


	file = open('test.json','w',encoding='utf-8')
	json.dump(final_json,file,ensure_ascii=False)

def yolo2json(val1_dir,jsonout_dir):
	model_yolo3 = Init_yolo("yolov3.weights", "cfg/yolov3.cfg")
	csv_reader = csv.reader(open("./data/val1_filename_id.csv"))
	file2id = {}
	final_json = {'results': [] }
	num = 0
	for row in csv_reader:
		file2id[row[0]] = row[1]
	for scene_num, scene in enumerate(sorted(os.listdir(val1_dir))):
		scene_dir = os.path.join(val1_dir,scene)
		for p, img_dir in enumerate(sorted(os.listdir(scene_dir))):
			print('第%d张图片'%num)
			final_json["results"].append({"image_id": file2id[img_dir], "object":[]})
			img_dir2 = os.path.join(scene_dir,img_dir)
			img = cv2.imread(img_dir2)
			# yolo_confidence =0.5 if scene_num < 3 else 0.3
			if scene_num == 0:
				confidence_yolo = 0.5
			elif scene_num == 1:
				confidence_yolo = 0.3
			elif scene_num == 2:
				confidence_yolo = 0.3
			elif scene_num == 3:
				confidence_yolo = 0.4
			elif scene_num == 4:
				confidence_yolo = 0.25
			outputs, ret = yolo3(model_yolo3, img,confidence_yolo, biggest_per=False)
			if ret :
				for i, output in enumerate(outputs):
					img_single = img[int(output[2]):int(output[4]),int(output[1]):int(output[3])]
					is_person = diff_scene_deal(img_single,scene_num)
					if is_person:
						final_json["results"][num]["object"].append({"minx": int(output[1]),
						                                             "miny": int(output[2]),
						                                             "maxx": int(output[3]),
						                                             "maxy": int(output[4]),
						                                             "staff": -1, "customer": -1, "stand": -1, "sit": -1,
						                                             "play_with_phone": -1, "male": -1,
						                                             "female": -1, "confidence": -1})
			num +=1

	with open(jsonout_dir,"w") as dump_f:
		json.dump(final_json, dump_f)
		print('save achieve')

def scene_test(img_dir,save_dir,scene_num):
	scene_num =scene_num-1
	model_yolo3 = Init_yolo("yolov3.weights", "cfg/yolov3.cfg")
	img = cv2.imread(img_dir)
	if scene_num == 0: confidence_yolo = 0.5
	elif scene_num == 1: confidence_yolo = 0.3
	elif scene_num == 2: confidence_yolo = 0.3
	elif scene_num == 3: confidence_yolo = 0.4
	elif scene_num == 4: confidence_yolo = 0.25

	outputs, ret = yolo3(model_yolo3, img, confidence_yolo, biggest_per=False)
	if ret:
		for i, output in enumerate(outputs):
			# print(i)
			img_single = img[int(output[2]):int(output[4]), int(output[1]):int(output[3])]
			is_person = diff_scene_deal(img_single, scene_num)
			if is_person:
				img = Image_rectangle(img,output[1:3],output[3:5],str(i))

			else:
				img = Image_rectangle(img,output[1:3],output[3:5],'mote')
			cv2.imwrite('./Test/' + str(i) + '.jpg', img_single)

	cv2.imwrite(save_dir,img)
	cv2.imshow('img',img)
	cv2.waitKey(0)

def diff_scene_deal(img,scene):
	# print(1)
	if img.shape[0]<50 and img.shape[1]<50:
		return False

	scene += 1
	if scene == 1:
		is_person = scene1(img)
	elif scene == 2:
		is_person = scene2(img)
	elif scene == 3:
		is_person = scene3(img)
	elif scene == 4:
		is_person = scene4(img)
	else :
		is_person = scene5(img)
	return is_person

def scene1(img):
	mianji = img.shape[0]*img.shape[1]
	if mianji <10000:
		return False
	else :
		return True

def scene2(img):
	return 1

def scene3(img):
	return 1

def scene4(img):
	average = junzhi(img)
	print('均值：',average)
	if average<23:
		return  True
	else:
		return False

def scene5(img):
	average = junzhi(img)
	print('均值：',average)
	if average<23:
		return  True
	else:
		return False

def junzhi(img):
	average = 0
	hight = img.shape[0]
	weight = img.shape[1]
	head_area = img[0:int(0.35*hight),int(0.2*weight):int(0.8*weight)]
	gray = cv2.cvtColor(head_area, cv2.COLOR_BGR2GRAY)
	ret, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
	gao, chang = binary.shape
	for l in range(gao):
		for j in range(chang):
				average += binary[l, j]
	average = average/(gao*chang)

	return average


if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '1'
	yolo2json('./data/val1',"./result/test_jw90_1.json")
	# scene_test('Test/scene_5_00325.jpg','Test/scene_5_00325-result.jpg',5)


