# import package
from glob import glob
# huawei
# import os
# os.system("pip install adabound -i http://repo.myhuaweicloud.com/repository/pypi/simple/")
import adabound
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import argparse
from tqdm import tqdm
import random
import json
import torch.nn.functional as F
# import file
from models.Resnet50MLP import myMLP
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
# import moxing as mox
# local_data_url = '/cache/dataset2/'
# local_train_url = '/cache/outputs/'

parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')

parser.add_argument('--model', type=str, default='models.resnet34-decoder.Net')
parser.add_argument('--stu_model', type=str, default='models.resnet18-decoder.Net')
parser.add_argument('--dataset_train', type=str, default='utils.dataset.DatasetForTrain')
parser.add_argument('--dataset_valid', type=str, default='utils.dataset.DatasetForValid')
parser.add_argument('--meta_train', type=str, default='meta/train/')
parser.add_argument('--meta_valid', type=str, default='meta/valid/')
parser.add_argument('--save-dir', type=str, default='./outputs/')
# parser.add_argument('--save-dir', type=str, default='/cache/outputs/') # huawei
parser.add_argument('--resume', type=str, default="")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--warmup-epochs', type=int, default=3)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--lr-min', type=float, default=1e-6)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--top-k', type=int, default=3)
parser.add_argument('--val-freq', type=int, default=2)
parser.add_argument('--pretrain', type=str, default="weights/resnet50-0676ba61.pth")
parser.add_argument('--pretrain2', type=str, default=r"D:\Users\11939\Desktop\Complex\Two-stage-Knowledge-For-Multiple-Adverse-Weather-Removal-main\weights\resnet18-f37072fd.pth")
parser.add_argument('--weight34', type=str, default=r"weights/resnet34")
parser.add_argument('--teachers', type=str,
                    default=["weights/fogbest.pkl", "weights/nightbest.pkl", "weights/rainbest.pkl",
                             "weights/snowbest.pkl"])
args = parser.parse_args()

writer = SummaryWriter(os.path.join(args.save_dir, 'log'))


# huawei
# if not os.path.exists(local_data_url):
#     os.makedirs(local_data_url)
# mox.file.copy_parallel(args.data_url, local_data_url)


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


def train_kc_stage(model, train_loader, optimizer, scheduler, epoch, criterions):
    print(Fore.CYAN + "==> Training Stage 1")
    print("==> Epoch {}/{}".format(epoch, args.max_epoch))
    print("==> Learning Rate = {:.6f}".format(optimizer.param_groups[0]['lr']))
    meters = get_meter(num_meters=5)

    criterion_l1, criterion_scr, criterion_hcr = criterions
    model.train()

    start = time.time()
    pBar = tqdm(train_loader, desc='Training')
    for target_images, input_images, tup in pBar:
        # Check whether the batch contains all types of degraded data
        if target_images is None:
            continue
        # move to GPU
        target_images = target_images.cuda()
        input_images = [images.cuda() for images in input_images]

        preds_from_student, features_from_student = model(torch.cat(input_images), return_feat=True)

        # Project the features to common feature space and calculate the loss
        # contra_loss = 0.
        # temperature = 0.5
        # # weights = [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # weights = [1.0 / 8, 1.0 / 4, 1.0]
        # for i, s_features in enumerate(features_from_student):
        #     # [B, D] 8 16 24 32
        #     out = mymlp[i](s_features)
        #     # [B, B]
        #     sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        #     mask = (torch.ones_like(sim_matrix) - torch.eye(args.batch_size, device=sim_matrix.device)).bool()
        #     # [B, B-1]
        #     sim_matrix = sim_matrix.masked_select(mask).view(args.batch_size, -1)
        #
        #     positive_sims = []
        #     idx = 0
        #     for j in range(4):
        #         edge = tup[j]
        #         # [B/4, B/4]
        #         cell = out[idx:idx+edge]
        #         # [B/4, B/4]
        #         cell_matrix = torch.exp(torch.mm(cell, cell.t().contiguous()) / temperature)
        #         cell_mask = (torch.ones_like(cell_matrix) - torch.eye(edge, device=cell_matrix.device)).bool()
        #         # [B/4, B/4-1]
        #         cell_matrix = cell_matrix.masked_select(cell_mask).view(edge, -1)
        #         # [B/4]
        #         positive_sim = torch.sum(cell_matrix, dim=-1)
        #         positive_sims.append(positive_sim)
        #         idx += edge
        #
        #     # [B]
        #     positive_sims = torch.cat(positive_sims, dim=0)
        #     # [B]
        #     total = sim_matrix.sum(dim=-1)
        #     loss = (- torch.log(positive_sims / (total + 1e-7) + 1e-7)).mean()
        #     contra_loss += loss * weights[i]
        #     # print(i, " contra_loss: ", loss.item())

        T_loss = criterion_l1(preds_from_student, target_images)
        SCR_loss = criterion_scr(preds_from_student, target_images, torch.cat(input_images))
        total_loss = T_loss + 0.1 * SCR_loss
        # print("Tot_contra_loss: ", contra_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        meters = update_meter(meters, [total_loss.item(), T_loss.item(), SCR_loss.item(),
                                       SCR_loss.item(), SCR_loss.item()])
        pBar.set_postfix({'loss': '{:.3f}'.format(meters[0].avg)})

    print("\nResults")
    print("------------------")
    print("Total loss: {:.3f}".format(meters[0].avg))
    print("------------------")
    print('Costing time: {:.3f}'.format((time.time() - start) / 60))
    print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    global writer
    writer.add_scalars('loss', {'train total loss': meters[0].avg}, epoch)
    writer.add_scalars('loss', {'train T loss': meters[1].avg}, epoch)
    writer.add_scalars('loss', {'train PFE loss': meters[2].avg}, epoch)
    writer.add_scalars('loss', {'train PFV loss': meters[3].avg}, epoch)
    writer.add_scalars('loss', {'train SCR loss': meters[4].avg}, epoch)

    writer.add_scalars('lr', {'Model lr': optimizer.param_groups[0]['lr']}, epoch)
    writer.add_scalars('lr', {'CKT lr': optimizer.param_groups[1]['lr']}, epoch)

    scheduler.step()


def compute_reprojection_loss(pred, target, ssim):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = ssim(pred, target).mean(1, True)
    # reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    reprojection_loss = ssim_loss

    return reprojection_loss


def train_ke_stage(model, train_loader, optimizer, scheduler, epoch, criterions, t_model):
    start = time.time()
    print(Fore.CYAN + "==> Training Stage2")
    print("==> Epoch {}/{}".format(epoch, args.max_epoch))
    print("==> Learning Rate = {:.6f}".format(optimizer.param_groups[0]['lr']))
    meters = get_meter(num_meters=3)

    ssim = SSIM()
    ssim.cuda()

    criterion_l1, criterion_scr, criterion_hcr = criterions

    model.train()
    t_model.eval()

    pBar = tqdm(train_loader, desc='Training')
    for target_images, input_images in pBar:

        # Check whether the batch contains all types of degraded data
        if target_images is None:
            continue

        # move to GPU
        target_images = target_images.cuda()
        input_images = torch.cat(input_images).cuda()

        preds, features = model(input_images, return_feat=True)
        t_preds, t_features = t_model(input_images, return_feat=True)

        G_loss = criterion_l1(preds, target_images)
        # G_loss += compute_reprojection_loss(preds, target_images, ssim).mean()

        F_loss = 0
        F_loss = criterion_l1(preds, t_preds)
        # F_weight = [1/8, 1/4, 1/2, 1]
        # for i in range(4):
        #     f_loss = criterion_l1(features[i], t_features[i])
        #     F_loss += F_weight[i] * f_loss

        # G_loss = 0
        # for i in range(4):
        #     G_loss += compute_reprojection_loss(features[i], target_images, ssim).mean()
        # G_loss /= 4

        HCR_loss = 1.0 * criterion_hcr(preds, target_images, input_images)
        total_loss = G_loss + HCR_loss + F_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        meters = update_meter(meters, [total_loss.item(), G_loss.item(), HCR_loss.item()])
        pBar.set_postfix({'loss': '{:.3f}'.format(meters[0].avg)})

    print("\nResults")
    print("------------------")
    print("Total loss: {:.3f}".format(meters[0].avg))
    print("------------------")
    print('Costing time: {:.3f}'.format((time.time() - start) / 60))
    print('Current time:', time.strftime("%H:%M:%S", time.localtime()))
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    global writer
    writer.add_scalars('loss', {'train total loss': meters[0].avg}, epoch)
    writer.add_scalars('loss', {'train G loss': meters[1].avg}, epoch)
    writer.add_scalars('loss', {'train HCR loss': meters[2].avg}, epoch)

    writer.add_scalars('lr', {'Model lr': optimizer.param_groups[0]['lr']}, epoch)

    scheduler.step()


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
    net_func_teacher = get_func(args.model)
    net_func_student = get_func(args.stu_model)
    dataset_train_func = get_func(args.dataset_train)
    dataset_valid_func = get_func(args.dataset_valid)
    print(Back.RED + 'Using Model: {}'.format(args.model) + Style.RESET_ALL)
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
                              drop_last=True, shuffle=True, collate_fn=Collate(n_degrades=4))
    val_loader = DataLoader(dataset=val_dataset, num_workers=args.num_workers, batch_size=1, drop_last=False,
                            shuffle=False)
    print(Style.BRIGHT + Fore.YELLOW + "# Training data / # Val data:" + Style.RESET_ALL)
    print(Style.BRIGHT + Fore.YELLOW + '{} / {}'.format(len(train_dataset), len(val_dataset)) + Style.RESET_ALL)
    print(Fore.RED + "---------------------------------------------------------------" + Style.RESET_ALL)

    # prepare the loss function
    criterions = nn.ModuleList([nn.L1Loss(), SCRLoss(args.pretrain), HCRLoss(args.pretrain)]).cuda()


    # Prepare the Model
    t_model = net_func_teacher(False).cuda()
    checkpoint34 = torch.load(args.weight34)
    t_model.load_state_dict(checkpoint34['state_dict'])
    model = net_func_student(True, args.pretrain2, 2).cuda()
    # model = Spectral_Normalization(model)
    # if args.isStageOne:
    #     mymlp = nn.ModuleList([])
    #     for c, d in zip([64, 128, 256, 256], [131072, 65536, 32768, 32768]):
    #         mymlp.append(myMLP(channel_t=c, channel_h=c // 2, feature_dim_o=d))
    #     mymlp = mymlp.cuda()
    # else:
    #     mymlp = nn.ModuleList([])
    #     for c, d in zip([64, 128, 256, 256], [128, 128, 128, 128]):
    #         mymlp.append(myMLP(channel_t=c, channel_h=c // 2, feature_dim_o=d))
    #     mymlp = mymlp.cuda()
    # mymlp1 = myMLP(262144).cuda()
    # mymlp2 = myMLP(131072).cuda()
    # mymlp3 = myMLP(65536).cuda()
    # mymlp4 = myMLP(65536).cuda()
    # mymlp = [mymlp1, mymlp2, mymlp3, mymlp4]

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        # mymlp.load_state_dict(checkpoint['mymlp'])
        # mymlp2.load_state_dict(checkpoint['mymlp2'])
        # mymlp3.load_state_dict(checkpoint['mymlp3'])
        # mymlp4.load_state_dict(checkpoint['mymlp4'])


    # prepare the optimizer and scheduler
    linear_scaled_lr = args.lr * args.batch_size / 16

    # if args.isStageOne:
    # optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': mymlp.parameters()}],
    #                          lr=linear_scaled_lr, betas=(0.9, 0.999), eps=1e-8)
    # else:
    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                             lr=linear_scaled_lr, betas=(0.9, 0.999), eps=1e-8)


    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch - args.warmup_epochs,
                                                                  eta_min=args.lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=args.warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    # Start training pipeline
    start_epoch = 1

    if args.resume:
        start_epoch = checkpoint['epoch'] + 1

    top_k_state = []
    print(Fore.GREEN + "Model would be saved on '{}'".format(args.save_dir) + Style.RESET_ALL)
    for epoch in range(start_epoch, args.max_epoch + 1):
        # training
        # if epoch <= 50:
        #     train_kc_stage(model, train_loader, optimizer, scheduler, epoch, criterions)
        # else:
        train_ke_stage(model, train_loader, optimizer, scheduler, epoch, criterions, t_model)
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
        # mox.file.copy_parallel(local_train_url, args.train_url)


if __name__ == '__main__':
    main()
