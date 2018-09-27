import os
import shutil
import json
import cv2

def make_testdata(train_dir,test_dir):
	if not os.path.isdir(test_dir):
		os.makedirs(test_dir)

	train_image = os.path.join(train_dir,'image')
	test_image = os.path.join(test_dir,'image')

	for i , train_image_sub in enumerate(os.listdir(train_image)):
		if i%10 == 0 :
			print('准备移动文件')
			image_dir = os.path.join(train_image,train_image_sub)
			new_image_dir = os.path.join(test_image,train_image_sub)
			print(image_dir)
			shutil.move(image_dir,new_image_dir)
			print('移动文件完成')
		print(train_image_sub,i)

	print('==================================')

	train_label = os.path.join(train_dir,'label')
	test_label = os.path.join(test_dir,'label')

	for i , train_label_sub in enumerate(os.listdir(train_label)):
		if i%10 == 0 :
			print('准备移动文件')
			label_dir = os.path.join(train_label,train_label_sub)
			new_label_dir = os.path.join(test_label,train_label_sub)
			print(label_dir)
			shutil.move(label_dir,new_label_dir)
		print(train_label_sub,i)

def crop_and_makelabel(images_dir,labels_dir,images_out_dir):
	if not os.path.isdir(images_out_dir):
		os.makedirs(images_out_dir)
	for image_dir in os.listdir(images_dir):
		label_dir = os.path.join(labels_dir, image_dir.split(".")[0] + ".json")
		image_dir2 = os.path.join(images_dir, image_dir)

		with open(label_dir, 'r') as load_f:
			load_dict = json.load(load_f)
			for i, person in enumerate(load_dict['annotation'][0]['object']):
				# position0 = (person['minx'], person['miny'])
				# position1 = ((person['maxx'], person['maxy']))
				img = cv2.imread(image_dir2)
				img_crop = img[person['miny']:person['maxy'], person['minx']:person['maxx']]
				name = image_dir.split('.')[0] + '_'+ str(i) + '_'+  str(person['gender']) + '_'+\
				       str(person['staff']) + '_'+ str(person['customer']) + '_'+ str(person['stand']) + '_'+ \
					   str(person['sit']) + '_' + str(person['play_with_phone']) + '.jpg'
				name_dir = os.path.join(images_out_dir,name)
				cv2.imwrite(name_dir,img_crop)
				print(name)


				# dateset.append((image_dir,
				#                 position0, position1,
				#                 person['gender'],
				#                 person['staff'],
				#                 person['customer'],
				#                 person['stand'],
				#                 person['sit'],
				#                 person['play_with_phone']
				#                 ))
'''
将训练数据标签转化为霞姐的faster-rccc格式
图片名字 

'''
def label2txt(input_dir,output_dir):
	num = 1
	with open(output_dir, 'w') as f:
		for labelpath in os.listdir(input_dir):
			labelpath = os.path.join(input_dir,labelpath)
			with open(labelpath,'r') as load_f:
				print('第%d张图片'%num)
				num += 1
				load_dict = json.load(load_f)
				filename = load_dict['annotation'][0]['filename']
				for i, person in enumerate(load_dict['annotation'][0]['object']):
					cls = person['name']
					minx = person['minx']
					miny = person['miny']
					maxx = person['maxx']
					maxy = person['maxy']

					message = filename + ' '+ cls + ' ' + str(minx) + ' ' + str(miny) + ' ' + str(maxx) + ' ' + str(maxy)
					print(message)
					f.write(str(message) + '\n')



		# i = '11.jpg'+' '+'person'
		# f.write(str(i))
	# with open (input_dir,'r') as f:
	# 	lines = f.readlines()
	# for line in lines:
	# 	print(line,type(line))
	# 	break


if __name__ == '__main__':
	# make_testdata('./data/train','./data/test')
	# crop_and_makelabel('./data/train/image','./data/train/label','./data/crop_train')
	# crop_and_makelabel('./data/test/image','./data/test/label','./data/crop_test')
	# label2txt('./data/label','data/bot_img.txt')
	with open('data/bot_img.txt', 'r') as f:
		lines = f.readlines()
		for line in lines:
			print(line,type(line))

