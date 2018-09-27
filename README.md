# kill-all

1.requirement：

  1）pytorch 0.4
  
  2)python3
  
2.data：
  mkdir data:
  
  ./data/train/image
  
  ./data/train/label
  
  ./data/test/image
  
  ./data/test/label
  
3.train：
  use the trian_mutli_task.py
  
  input command such as : 
  
  python trian_mutli_task.py -a ResNet50_BOT_MultiTask -d BOT
  
  best multi-task-model:ResNet50_BOT_MultiTask(defined in ./models/ResNet.py)
  
4.recognition:

  multi-task-demo.py 
  
  most important function:get_result()
  
  this function load the traind model to identify person attributes
  
  
5.person detction:

  use yolo3 without train or use yolo2 which traind in this dataset(https://github.com/jianwu585218/yolov2)
