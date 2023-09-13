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
import torchvision.transforms.functional as TF
# import file
from utils.utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='models.resnet18-decoder.Net')
parser.add_argument('--dataset', type=str, default='utils.dataset.DatasetForInference')
parser.add_argument('--checkpoint', type=str, default='./weights/epoch_257_psnr28.605_ssim0.917')
parser.add_argument('--dir_path', type=str, default='Traffic-sign-data/val_train')
parser.add_argument('--save_dir', type=str, default='Traffic-sign-data/outputs_train')
parser.add_argument('--num_workers', type=int, default=0)
args = parser.parse_args()


# @torch.no_grad()
# def evaluate(model, loader):
#
# 	print(Fore.GREEN + "==> Inference")
#
# 	start = time.time()
# 	model.eval()
# 	fps = 0.0  # 计算帧数
# 	cnt = 0
# 	for imSnow, image_name in tqdm(loader, desc='Inference'):
# 		H = imSnow.shape[2]
# 		W = imSnow.shape[3]
# 		patchSize = 64
# 		deSnowImage = torch.zeros(imSnow.shape)
# 		deSnowImage = deSnowImage.cpu()
# 		for i in range(0, W - 64, patchSize):
# 			for j in range(0, H - 64, patchSize):
# 				torch.cuda.empty_cache()
# 				croppedI = TF.crop(imSnow, j, i, patchSize, patchSize)
# 				# Run the model
# 				croppedI = croppedI.cuda()
# 				y_hat, _ = model(croppedI)
# 				torch.cuda.synchronize()
# 				y_hat = y_hat.cpu()
# 				deSnowImage[:, :, j:j + patchSize, i:i + patchSize] = y_hat
# 				torch.cuda.synchronize()
# 				croppedI = croppedI.to("cpu")
#
# 		file_name = os.path.join(args.save_dir, image_name[0])
# 		torchvision.utils.save_image(deSnowImage.cpu(), file_name)
# 	print('Costing time: {:.3f}'.format((time.time()-start)/60))
# 	print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
# 	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)
# 	print(fps)


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


@torch.no_grad()
def evaluate(model, loader):

	print(Fore.GREEN + "==> Inference")

	start = time.time()
	model.eval()
	fps = 0.0  # 计算帧数
	cnt = 0
	for image, image_name, delt in tqdm(loader, desc='Inference'):

		if torch.cuda.is_available():
			image = image.cuda()
		t1 = time_sync()
		pred = model(image)
		fps = (fps + (1. / (time_sync() - t1 + delt + 1e-7))) / 2  # 计算平均fps
		file_name = os.path.join(args.save_dir, image_name[0])
		torchvision.utils.save_image(pred.cpu(), file_name)
		# cnt += 1
		# if cnt == 1000:
		# 	break
	print('Costing time: {:.3f}'.format((time.time()-start)/60))
	print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)
	print(fps)


def main():

	# get the net and dataset function
	net_func = get_func(args.model)
	dataset_func = get_func(args.dataset)
	print(Back.RED + 'Using Model: {}'.format(args.model) + Style.RESET_ALL)
	print(Back.RED + 'Using Dataset: {}'.format(args.dataset) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# prepare the dataloader
	dataset = dataset_func(dir_path=args.dir_path)
	loader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, drop_last=False, shuffle=False, pin_memory=True)
	print(Style.BRIGHT + Fore.YELLOW + "# Val data: {}".format(len(dataset)) + Style.RESET_ALL)
	print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)


	# prepare the model
	model = net_func()
	model = Spectral_Normalization(model)

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
