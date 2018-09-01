import json

with open('scene_1_00002.json','r') as load_f:
	load_dict = json.load(load_f)
	print(load_dict)
	print(type(load_dict))
