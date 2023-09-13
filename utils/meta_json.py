import os
import json
import random

current_data_index_list = list(range(6419))
random.shuffle(current_data_index_list)
current_data_index_list = current_data_index_list[:500]

file = "../meta-all/train/fog_train.json"
file2 = "./fog_train.json"
file_in = open(file, "r")
json_data = json.load(file_in)
json_data2 = []
file_out = open(file2, "w")

for idx, line in enumerate(json_data):
    if idx in current_data_index_list:
        json_data2.append(line)

file_out.write(json.dumps(json_data2))
file_in.close()
file_out.close()










