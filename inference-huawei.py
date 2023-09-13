# import package

import sys
from glob import glob
from pathlib import Path
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import numpy as np
import torch
import torchvision
import time
from torch.utils.data import DataLoader
from colorama import Style, Fore, Back
import argparse
from tqdm import tqdm
import torchvision.transforms.functional as TF
# import file
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')

parser.add_argument('--model', type=str, default='models.DeHaze.Net')
parser.add_argument('--dataset', type=str, default='utils.dataset.DatasetForInference2')
parser.add_argument('--checkpoint', type=str, default='./weights/Desnow') # huawei
parser.add_argument('--dir_path', type=str, default='Traffic-sign-data/val_snow')
parser.add_argument('--save_dir', type=str, default='Traffic-sign-data/outputs_snow')
# parser.add_argument('--save_dir', type=str, default='/cache/outputs/') # huawei
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--meta_train', type=str, default='meta/train/')
parser.add_argument('--meta_valid', type=str, default='meta/valid/')
args = parser.parse_args()

# huawei
# import moxing as mox
# local_data_url = '/cache/dataset2/'
# local_train_url = '/cache/outputs/'
# if not os.path.exists(local_data_url):
#     os.makedirs(local_data_url)
# if not os.path.exists(local_train_url):
#     os.makedirs(local_train_url)
# mox.file.copy_parallel(args.data_url, local_data_url)


@torch.no_grad()
def evaluate(model, loader):
	print(Fore.GREEN + "==> Inference")
	start = time.time()
	model.eval()
	for imSnow, image_name in tqdm(loader, desc='Inference'):
		H = imSnow.shape[2]
		W = imSnow.shape[3]
		patchSize = 64
		deSnowImage = torch.zeros(imSnow.shape)
		deSnowImage = deSnowImage.cpu()
		for i in range(0, W - 64, patchSize):
			for j in range(0, H - 64, patchSize):
				torch.cuda.empty_cache()
				croppedI = TF.crop(imSnow, j, i, patchSize, patchSize)
				# Run the model
				croppedI = croppedI.cuda()
				y_hat, y_dash, z_hat, a = model(croppedI)
				torch.cuda.synchronize()
				y_hat = y_hat.cpu()
				deSnowImage[:, :, j:j + patchSize, i:i + patchSize] = y_hat
				torch.cuda.synchronize()
				croppedI = croppedI.to("cpu")
		file_name = os.path.join(args.save_dir, image_name[0])
		torchvision.utils.save_image(deSnowImage.cpu(), file_name)
	print('Costing time: {:.3f}'.format((time.time() - start) / 60))
	print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	# huawei
	# mox.file.copy_parallel(local_train_url, args.train_url)


# @torch.no_grad()
# def evaluate(model, loader):
#
# 	print(Fore.GREEN + "==> Inference")
# 	start = time.time()
# 	model.eval()
# 	for image, image_name in tqdm(loader, desc='Inference'):
# 		if torch.cuda.is_available():
# 			image = image.cuda()
# 		pred, y_, z_hat, a = model(image)
# 		file_name = os.path.join(args.save_dir, image_name[0])
# 		torchvision.utils.save_image(pred.cpu(), file_name)
# 	print('Costing time: {:.3f}'.format((time.time() - start) / 60))
# 	print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
# 	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	# huawei
	# mox.file.copy_parallel(local_train_url, args.train_url)


def main():

	# get the net and dataset function
	net_func = get_func(args.model)
	dataset_func = get_func(args.dataset)
	print(Back.RED + 'Using Model: {}'.format(args.model) + Style.RESET_ALL)
	print(Back.RED + 'Using Dataset: {}'.format(args.dataset) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	# meta_train_paths = sorted(glob(os.path.join(ROOT / args.meta_train, '*.json')))
	meta_valid_paths = sorted(glob(os.path.join(ROOT / args.meta_valid, '*.json')))

	# prepare the dataloader
	dataset = dataset_func(meta_valid_paths)
	loader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, drop_last=False, shuffle=False, pin_memory=True)
	print(Style.BRIGHT + Fore.YELLOW + "# Val data: {}".format(len(dataset)) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

	# prepare the model
	model = net_func()
	# model = Spectral_Normalization(model)

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

	evaluate(model, loader)

if __name__ == '__main__':
	main()
