# import package
import numpy as np
import torch
import torchvision
import time
import os
from torch.utils.data import DataLoader
from colorama import Style, Fore, Back
import argparse
from tqdm import tqdm
import cv2

# import file
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='models.mobilenetv2-small.Net')
parser.add_argument('--dataset', type=str, default='utils.dataset.DatasetForInference')
parser.add_argument('--checkpoint', type=str, default='./weights/mobilenetv2-small')
parser.add_argument('--dir_path', type=str, default='./Traffic-sign-data/val_night')
parser.add_argument('--save_dir', type=str, default='./Traffic-sign-data/outputs_night')
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()


def main():

	# get the net and dataset function
	net_func = get_func(args.model)
	dataset_func = get_func(args.dataset)
	print(Back.RED + 'Using Model: {}'.format(args.model) + Style.RESET_ALL)
	print(Back.RED + 'Using Dataset: {}'.format(args.dataset) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	# prepare the model
	model = net_func()

	# load the checkpoint
	assert os.path.isfile(args.checkpoint), "The checkpoint '{}' does not exist".format(args.checkpoint)
	checkpoint = torch.load(args.checkpoint)
	msg = model.load_state_dict(checkpoint["state_dict"], strict=False)

	print(Fore.GREEN + "Loaded checkpoint from '{}'".format(args.checkpoint) + Style.RESET_ALL)
	print(Fore.GREEN + "{}".format(msg) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	# move to GPU
	if torch.cuda.is_available():
		model = model.cuda()
		model.eval()
		# model.half()

	with torch.no_grad():

		fps = 0.0  # 计算帧数
		cnt = 0
		from torchvision import transforms
		transform = transforms.ToTensor()
		img_dir = os.listdir(args.dir_path)
		for img_path in img_dir:
			t1 = time.time()
			image = os.path.join(args.dir_path, img_path)
			image = cv2.imread(image)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = image[:, :, ::-1].transpose(2, 0, 1)
			image = np.ascontiguousarray(image)
			input_image = torch.from_numpy(image).cuda()
			_, h, w = input_image.shape
			if (h % 16 != 0) or (w % 16 != 0):
				input_image = transforms.Resize(((h // 16) * 16, (w // 16) * 16))(input_image)
			if input_image.ndimension() == 3:
				input_image = input_image.unsqueeze(0)
			input_image = input_image.float()

			pred = model(input_image)
			fps = (fps + (1. / (time.time() - t1 + 1e-3))) / 2  # 计算平均fps
			cnt += 1
			if cnt == 500:
				break
			print(cnt)

		print(fps)

if __name__ == '__main__':
	main()
