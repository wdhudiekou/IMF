
import sys
sys.path.append("..")
import pathlib
import warnings
import logging.config

import numpy
import torch.backends.cudnn
import torch.utils.data
import torchvision.transforms
import visdom
from tqdm import tqdm
import torch.nn.functional
from dataloader.fuse_data_vsm import FuseDataVSM
from models.co_fusion_net import CoFusionNet
from loss.fusion_SA_loss import FusionSALoss
import argparse, os

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RobF Net train process')

    # RoadScene dataset
    parser.add_argument('--ir_reg', default='../registered_ir_path/RoadScene/ir_reg', type=pathlib.Path)
    parser.add_argument('--vi',     default='../dataset/raw/ctrain/RoadScene/vi', type=pathlib.Path)
    parser.add_argument('--ir_map', default='../dataset/raw/ctrain/RoadScene/ir_map_soft', type=pathlib.Path)
    parser.add_argument('--vi_map', default='../dataset/raw/ctrain/RoadScene/vi_map_soft', type=pathlib.Path)

    # MSIFT dataset
    # parser.add_argument('--ir_reg', default='../registered_ir_path/MSIFT/ir_reg', type=pathlib.Path)
    # parser.add_argument('--vi',     default='../dataset/raw/ctrain/MSIFT/vi_s', type=pathlib.Path)
    # parser.add_argument('--ir_map', default='../dataset/raw/ctrain/MSIFT/ir_map_soft_s', type=pathlib.Path)
    # parser.add_argument('--vi_map', default='../dataset/raw/ctrain/MSIFT/vi_map_soft_s', type=pathlib.Path)

    # M3FD dataset
    # parser.add_argument('--ir_reg', default='../registered_ir_path/M3FD/ir_reg', type=pathlib.Path)
    # parser.add_argument('--vi',     default='../dataset/raw/ctrain/M3FD/vi', type=pathlib.Path)
    # parser.add_argument('--ir_map', default='../dataset/raw/ctrain/M3FD/ir_map_soft', type=pathlib.Path)
    # parser.add_argument('--vi_map', default='../dataset/raw/ctrain/M3FD/vi_map_soft', type=pathlib.Path)

    # FLIR dataset
    # parser.add_argument('--ir_reg', default='../dataset/raw/ctrain/FLIR/ir_reg_duse', type=pathlib.Path)
    # parser.add_argument('--vi',     default='../dataset/raw/ctrain/FLIR/vi', type=pathlib.Path)
    # parser.add_argument('--ir_map', default='../dataset/raw/ctrain/FLIR/ir_map_soft', type=pathlib.Path)
    # parser.add_argument('--vi_map', default='../dataset/raw/ctrain/FLIR/vi_map_soft', type=pathlib.Path)


    # train loss weights
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=20.0, type=float)
    parser.add_argument('--theta', default=5.0, type=float)
    # implement details
    parser.add_argument('--dim', default=64, type=int, help='AFuse feather dim')
    parser.add_argument('--batchsize', default=8, type=int, help='mini-batch size')  # 32
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument("--step", type=int, default=1000, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
    parser.add_argument('--resume', default='', help='resume checkpoint')
    parser.add_argument('--interval', default=20, help='record interval')
    # checkpoint
    parser.add_argument("--load_model_fuse", default=None, help="path to pretrained model (default: none)")
    # please replace 'checkpoint_path' with yourself filename
    parser.add_argument('--ckpt', default='../cache/Fusion_only/Road/checkpoint_path/', help='checkpoint cache folder')

    args = parser.parse_args()
    return args

def main(args, visdom):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    log = logging.getLogger()

    epoch = args.nEpochs
    interval = args.interval

    print("===> Creating Save Path of Checkpoints")
    cache = pathlib.Path(args.ckpt)

    print("===> Loading datasets")
    crop = torchvision.transforms.RandomResizedCrop((256, 256))
    data = FuseDataVSM(args.ir_reg, args.vi, args.ir_map, args.vi_map, crop)
    training_data_loader = torch.utils.data.DataLoader(data, args.batchsize, True, pin_memory=True)

    print("===> Building models")
    FuseNet = CoFusionNet(args.dim).to(device)

    print("===> Defining Loss fuctions")
    criterion_fus = FusionSALoss(args.alpha, args.beta, args.theta).to(device)

    print("===> Setting Optimizers")
    optimizer_fus = torch.optim.Adam(params=FuseNet.parameters(), lr=args.lr)

    # TODO: optionally copy weights from a checkpoint
    if args.load_model_fuse is not None:
        print('Loading pre-trained FuseNet checkpoint %s' % args.load_model_fuse)
        log.info(f'Loading pre-trained checkpoint {str(args.load_model_fuse)}')
        state = torch.load(str(args.load_model_fuse))
        FuseNet.load_state_dict(state['net'])
    else:
        print("=> no model found at '{}'".format(args.load_model_fuse))

    print("===> Starting Training")
    for epoch in range(args.start_epoch, args.nEpochs + 1):
        train(args, training_data_loader, optimizer_fus, FuseNet, criterion_fus, epoch)

        # TODO: save checkpoint
        save_checkpoint(FuseNet, epoch, cache) if epoch % interval == 0 else None

def train(args, training_data_loader, optimizer_fus, FuseNet, criterion_fus, epoch):

    FuseNet.train()
    # TODO: update learning rate of the optimizer
    lr_F = adjust_learning_rate(args, optimizer_fus, epoch - 1)
    print("Epoch={}, lr_F={} ".format(epoch, lr_F))
    tqdm_loader = tqdm(training_data_loader, disable=True)
    num = len(training_data_loader)
    step = 0.
    loss_total, loss_reg, loss_fus = [], [], []
    for (ir, vi), _, (ir_map, vi_map)in tqdm_loader:
        step += 1.0 / num
        ir_reg, vi         = ir.cuda(), vi.cuda()
        ir_map, vi_map = ir_map.cuda(), vi_map.cuda()

        fuse_out  = FuseNet(ir_reg, vi)
        loss = criterion_fus(fuse_out, ir_reg, vi, ir_map, vi_map, step)

        optimizer_fus.zero_grad()
        loss.backward()
        optimizer_fus.step()

        if tqdm_loader.n % 40 == 0:
            show = torch.stack([ir_reg[0], vi[0], fuse_out[0]])
            visdom.images(show, win='Fusion')

        loss_total.append(loss.item())
    loss_avg = numpy.mean(loss_total)
    # TODO: visdom display
    visdom.line([loss_avg], [epoch], win='loss-Fusion', name='total', opts=dict(title='Total-loss'), update='append' if epoch else '')



def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.step))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def save_checkpoint(net, epoch, cache):
    model_folder = cache
    model_out_path = str(model_folder / f'fus_{epoch:04d}.pth')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    args = hyper_args()
    visdom = visdom.Visdom(port=8097, env='Fusion')

    main(args, visdom)