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
# import moxing as mox
# local_data_url = '/cache/dataset2/'
# local_train_url = '/cache/outputs/'

parser = argparse.ArgumentParser()
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')

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


def compute_reprojection_loss(pred, target, ssim):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = ssim(pred, target).mean(1, True)
    # reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    reprojection_loss = ssim_loss

    return reprojection_loss


def train_ke_stage(model, train_loader, optimizer, scheduler, epoch):
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

        preds, features = model(input_images)


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
    model = net_func_student(True, args.pretrain2, 2).cuda()

    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    # prepare the optimizer and scheduler
    linear_scaled_lr = args.lr * args.batch_size / 16

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
        train_ke_stage(model, train_loader, optimizer, scheduler, epoch)
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
