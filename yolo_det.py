import cv2
from camera import *

'''
给图片加框和名字
'''
def Image_rectangle(img,c1,c2,name):
	c1 = tuple(c1.int())
	c2 = tuple(c2.int())
	color = (0, 0, 255)
	cv2.rectangle(img, c1, c2, color,1)    # 加框
	t_size = cv2.getTextSize(str(name), cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
	c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
	cv2.rectangle(img, c1, c2, color, -1) # -1填充作为文字框底色
	cv2.putText(img, str(name), (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
	return img

if __name__ == '__main__':
	model_yolo3 = Init_yolo("yolov3.weights","cfg/yolov3.cfg")
	img = cv2.imread('./bot9.jpg')
	# outputs, haveperson = yolo3(model_yolo3, img, biggest_per=False)
	outputs, ret = yolo3(model_yolo3, img, confidence=0.1, biggest_per=False)

	if ret:
		for i,output in enumerate(outputs):
			print(i,output)

			img_single = img[int(output[2]):int(output[4]), int(output[1]):int(output[3])]
			cv2.imshow('single',img_single)
			cv2.imshow('img',img)
			cv2.imwrite('./single.jpg',img_single)
			cv2.waitKey(0)

			img = Image_rectangle(img,output[1:3],output[3:5],str(i))

	else :
		print("Yolo3 can not detect person")

	cv2.imwrite('bot9-result-01.jpg',img)
	cv2.imshow('result',img)
	cv2.waitKey(0)

