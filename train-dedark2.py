# import package
from glob import glob
import time
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
from tqdm import tqdm
import random
import json
from models.SSIM_L1 import SSIM
from utils.dataset import Collate
from models.MCR import SCRLoss, HCRLoss
from utils.averageMeter import *
from utils.warmup_scheduler import GradualWarmupScheduler
from utils.utils import *
from utils import pytorch_ssim

from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# huawei
import moxing as mox
local_data_url = '/cache/dataset2/'
local_train_url = '/cache/outputs/'

parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')

parser.add_argument('--stu_model', type=str, default='models.ZeroDCE.DCENet')
parser.add_argument('--dataset_train', type=str, default='utils.dataset.DatasetForTrain')
parser.add_argument('--dataset_valid', type=str, default='utils.dataset.DatasetForValid')
parser.add_argument('--meta_train', type=str, default='meta/train/')
parser.add_argument('--meta_valid', type=str, default='meta/valid/')
# parser.add_argument('--save-dir', type=str, default='./outputs/')
parser.add_argument('--save-dir', type=str, default='/cache/outputs/') # huawei
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--warmup-epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr-min', type=float, default=1e-6)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--top-k', type=int, default=3)
parser.add_argument('--val-freq', type=int, default=2)
parser.add_argument('--weights', type=float, default=[8, 1.75, 1, 7],
                        help=('A list of weights for [w_spa, w_exp, w_col, w_tvA]. '
                              'That is, spatial loss, exposure loss, color constancy, '
                              'and total variation respectively'))
parser.add_argument('--loss', type=int, default=1, choices=[1, 2])

args = parser.parse_args()

writer = SummaryWriter(os.path.join(args.save_dir, 'log'))

# huawei
if not os.path.exists(local_data_url):
    os.makedirs(local_data_url)
mox.file.copy_parallel(args.data_url, local_data_url)


@torch.no_grad()
def evaluate(model, val_loader, epoch):
    print(Fore.GREEN + "==> Evaluating")
    print("==> Epoch {}/{}".format(epoch, args.max_epoch))

    psnr_list, ssim_list = [], []
    model.eval()
    start = time.time()
    pBar = tqdm(val_loader, desc='Evaluating')
    for target, image in pBar:

        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()

        pred = model(image)

        psnr_list.append(torchPSNR(pred, target).item())
        ssim_list.append(pytorch_ssim.ssim(pred, target).item())

    print("\nResults")
    print("------------------")
    print("PSNR: {:.3f}".format(np.mean(psnr_list)))
    print("SSIM: {:.3f}".format(np.mean(ssim_list)))
    print("------------------")
    print('Costing time: {:.3f}'.format((time.time() - start) / 60))
    print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    global writer
    writer.add_scalars('PSNR', {'val psnr': np.mean(psnr_list)}, epoch)
    writer.add_scalars('SSIM', {'val ssim': np.mean(ssim_list)}, epoch)

    return np.mean(psnr_list), np.mean(ssim_list)

import torch
import torch.nn.functional as F
def spatial_consistency_loss(enhances, originals, to_gray, neigh_diff, rsize=4):
    # convert to gray
    enh_gray = F.conv2d(enhances, to_gray)
    ori_gray = F.conv2d(originals, to_gray)

    # average intensity of local regision
    enh_avg = F.avg_pool2d(enh_gray, rsize)
    ori_avg = F.avg_pool2d(ori_gray, rsize)

    # calculate spatial consistency loss via convolution
    enh_pad = F.pad(enh_avg, (1, 1, 1, 1), mode='replicate')
    ori_pad = F.pad(ori_avg, (1, 1, 1, 1), mode='replicate')
    enh_diff = F.conv2d(enh_pad, neigh_diff)
    ori_diff = F.conv2d(ori_pad, neigh_diff)

    spa_loss = torch.pow((enh_diff - ori_diff), 2).sum(1).mean()
    return spa_loss

def exposure_control_loss(enhances, rsize=16, E=0.6):
    avg_intensity = F.avg_pool2d(enhances, rsize).mean(1)  # to gray: (R+G+B)/3
    exp_loss = (avg_intensity - E).abs().mean()
    return exp_loss

def color_constency_loss(enhances):
    plane_avg = enhances.mean((2, 3))
    col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                          + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                          + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
    return col_loss

def color_constency_loss2(enhances, originals):
    enh_cols = enhances.mean((2, 3))
    ori_cols = originals.mean((2, 3))
    rg_ratio = (enh_cols[:, 0] / enh_cols[:, 1] - ori_cols[:, 0] / ori_cols[:, 1]).abs()
    gb_ratio = (enh_cols[:, 1] / enh_cols[:, 2] - ori_cols[:, 1] / ori_cols[:, 2]).abs()
    br_ratio = (enh_cols[:, 2] / enh_cols[:, 0] - ori_cols[:, 2] / ori_cols[:, 0]).abs()
    col_loss = (rg_ratio + gb_ratio + br_ratio).mean()
    return col_loss

def alpha_total_variation(A):
    '''
    Links: https://remi.flamary.com/demos/proxtv.html
           https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html#total_variation
    '''
    delta_h = A[:, :, 1:, :] - A[:, :, :-1, :]
    delta_w = A[:, :, :, 1:] - A[:, :, :, :-1]

    # TV used here: L-1 norm, sum R,G,B independently
    # Other variation of TV loss can be found by google search
    tv = delta_h.abs().mean((2, 3)) + delta_w.abs().mean((2, 3))
    loss = torch.mean(tv.sum(1) / (A.shape[1] / 3))
    return loss



def train_ke_stage(model, train_loader, optimizer, scheduler, epoch, **kwargs):
    w_spa, w_exp, w_col, w_tvA = kwargs['w_spa'], kwargs['w_exp'], kwargs['w_col'], kwargs['w_tvA']
    to_gray, neigh_diff = kwargs['to_gray'], kwargs['neigh_diff']
    spa_rsize, exp_rsize = kwargs['spa_rsize'], kwargs['exp_rsize']

    start = time.time()
    print(Fore.CYAN + "==> Training Stage2")
    print("==> Epoch {}/{}".format(epoch, args.max_epoch))
    print("==> Learning Rate = {:.6f}".format(optimizer.param_groups[0]['lr']))
    meters = get_meter(num_meters=3)

    model.train()

    pBar = tqdm(train_loader, desc='Training')
    for target_images, input_images in pBar:

        # Check whether the batch contains all types of degraded data
        if target_images is None:
            continue

        # move to GPU
        target_images = target_images.cuda()
        input_images = torch.cat(input_images).cuda()

        results, Astack = model(input_images)
        enhanced_batch = results[-1]

        L_spa = w_spa * spatial_consistency_loss(
            enhanced_batch, input_images, to_gray, neigh_diff, spa_rsize)
        L_exp = w_exp * exposure_control_loss(enhanced_batch, exp_rsize, E=0.62)
        if args.loss == 1:
            L_col = w_col * color_constency_loss(enhanced_batch)
        elif args.loss == 2:
            L_col = w_col * color_constency_loss2(enhanced_batch, input_images)
        L_tvA = w_tvA * alpha_total_variation(Astack)
        loss = L_spa + L_exp + L_col + L_tvA

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        meters = update_meter(meters, [loss.item(), loss.item(), loss.item()])
        pBar.set_postfix({'loss': '{:.3f}'.format(meters[0].avg)})

    print("\nResults")
    print("------------------")
    print("Total loss: {:.3f}".format(meters[0].avg))
    print("------------------")
    print('Costing time: {:.3f}'.format((time.time() - start) / 60))
    print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    scheduler.step(loss)


def main():
    # Set up random seed
    random_seed = 19870522
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(Back.WHITE + 'Random Seed: {}'.format(random_seed) + Style.RESET_ALL)
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    # tensorboard
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # get the net and datasets function
    net_func_student = get_func(args.stu_model)
    dataset_train_func = get_func(args.dataset_train)
    dataset_valid_func = get_func(args.dataset_valid)
    print(Back.RED + 'Using Dataset for Train: {}'.format(args.dataset_train) + Style.RESET_ALL)
    print(Back.RED + 'Using Dataset for Valid: {}'.format(args.dataset_valid) + Style.RESET_ALL)
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    # load meta files
    meta_train_paths = sorted(glob(os.path.join(ROOT / args.meta_train, '*.json')))
    meta_valid_paths = sorted(glob(os.path.join(ROOT / args.meta_valid, '*.json')))

    # prepare the dataloader
    train_dataset = dataset_train_func(meta_paths=meta_train_paths)
    val_dataset = dataset_valid_func(meta_paths=meta_valid_paths)
    train_loader = DataLoader(dataset=train_dataset, num_workers=args.num_workers, batch_size=args.batch_size,
                              drop_last=True, shuffle=True, collate_fn=Collate(n_degrades=1))
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_workers, batch_size=1, drop_last=False,
                            shuffle=False)
    print(Style.BRIGHT + Fore.YELLOW + "# Training data / # Val data:" + Style.RESET_ALL)
    print(Style.BRIGHT + Fore.YELLOW + '{} / {}'.format(len(train_dataset), len(val_dataset)) + Style.RESET_ALL)
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    # prepare the loss function
    # criterions = nn.ModuleList([nn.L1Loss(), SCRLoss(args.pretrain), HCRLoss(args.pretrain)]).cuda()

    # Prepare the Model
    model = net_func_student(n=8, return_results=[4, 6, 8]).cuda()

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    w_spa, w_exp, w_col, w_tvA = args.weights
    hp = dict(lr=1e-4, wd=0, lr_decay_factor=0.97,
              w_spa=w_spa, w_exp=w_exp, w_col=w_col, w_tvA=w_tvA,
              spa_rsize=4, exp_rsize=16)

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=hp['lr'], weight_decay=hp['wd'])

    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, mode='min', factor=hp['lr_decay_factor'], threshold=3e-4)

    def get_kernels():
        # weighted RGB to gray
        K1 = torch.tensor([0.3, 0.59, 0.1], dtype=torch.float32).view(1, 3, 1, 1).cuda()
        # K1 = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float32).view(1, 3, 1, 1).to(device)

        # kernel for neighbor diff
        K2 = torch.tensor([[[0, -1, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
                           [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
                           [[0, 0, 0], [0, 1, -1], [0, 0, 0]]], dtype=torch.float32)
        K2 = K2.unsqueeze(1).cuda()
        return K1, K2

    to_gray, neigh_diff = get_kernels()  # conv kernels for calculating spatial consistency loss

    # Start training pipeline
    start_epoch = 1

    if args.resume:
        start_epoch = checkpoint['epoch'] + 1

    top_k_state = []
    print(Fore.GREEN + "Model would be saved on '{}'".format(args.save_dir) + Style.RESET_ALL)
    for epoch in range(start_epoch, args.max_epoch + 1):
        train_ke_stage(model, train_loader, optimizer, scheduler, epoch, to_gray=to_gray, neigh_diff=neigh_diff,
                     w_spa=hp['w_spa'], w_exp=hp['w_exp'], w_col=hp['w_col'], w_tvA=hp['w_tvA'],
                     spa_rsize=hp['spa_rsize'], exp_rsize=hp['exp_rsize'])
        # validating
        if epoch % args.val_freq == 0:
            psnr, ssim = evaluate(model, val_loader, epoch)
            # Check whether the model is top-k model
            top_k_state = save_top_k(model, optimizer, scheduler, top_k_state, args.top_k, epoch, args.save_dir, psnr=psnr, ssim=ssim)

        torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                   os.path.join(args.save_dir, 'latest_model'))

        # huawei
        mox.file.copy_parallel(local_train_url, args.train_url)


if __name__ == '__main__':
    main()
