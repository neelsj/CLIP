import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import Subset

import clip
import json
import os

from typing import Any, Callable, Optional, Tuple, List
from PIL import Image

from pycocotools.coco import COCO

class CocoDetectionCaption(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        annCapFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.coco = COCO(annFile)
        self.cocoCap = COCO(annCapFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

        self.coco_id_to_ind = {}
        for i, cat in enumerate(self.coco.cats):
            self.coco_id_to_ind[cat] = i

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(id))
        annsCap = self.cocoCap.loadAnns(self.cocoCap.getAnnIds(id))

        categories = [ann["category_id"] for ann in anns]
        categories = sorted(list(set(categories)))
        random.shuffle(categories)

        categories_text = [self.coco.cats[category_id]["name"] for category_id in categories]
        
        #print(categories_text)

        classes_prompt = f"A photo of a %s" % categories_text[0]
        
        if (len(categories_text) > 0):
            classes_prompt = classes_prompt + "".join([f", {c}" for c in categories_text[1:-1]])

        classes_prompt = classes_prompt + f", and a %s." % categories_text[-1]

        captions = [ann["caption"] for ann in annsCap]

        caption = random.choice(captions)

        #print(classes_prompt)
        #print(caption)
        
        classes_prompt_tokens = torch.squeeze(clip.tokenize(classes_prompt))
        caption_tokens = torch.squeeze(clip.tokenize(caption))
        
        return classes_prompt_tokens, caption_tokens

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        classes_prompt_tokens, caption_tokens = self._load_target(id)

        if self.transforms is not None:
            image, _ = self.transforms(image, None)

        return image, classes_prompt_tokens, caption_tokens

    def __len__(self) -> int:
        return len(self.ids)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', default='coco',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/32',
                    choices=clip.available_models(),
                    help='model architecture: ' +
                        ' | '.join(clip.available_models()) +
                        ' (default: ViT-B/32)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--mode', default='clip', choices=['clip', 'lin', 'context', 'lin_t'])
parser.add_argument('--classes', default=1000, type=int)

best_acc1 = 0


def main():
    args = parser.parse_args()

    if "AMLT_OUTPUT_DIR" in os.environ:
        print('output dir %s' % os.environ["AMLT_OUTPUT_DIR"])

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(args.arch, device, mode=args.mode, classes=args.classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs of the current node.
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CosineSimilarity().cuda(args.gpu)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train2017')
    valdir = os.path.join(args.data, 'val2017')

    trainjson = os.path.join(args.data, 'annotations/instances_train2017.json')
    traincapjson = os.path.join(args.data, 'annotations/captions_train2017.json')
    valjson = os.path.join(args.data, 'annotations/instances_val2017.json')
    valcapjson = os.path.join(args.data, 'annotations/captions_val2017.json')

    train_dataset = CocoDetectionCaption(
        traindir,
        trainjson,
        traincapjson,
        transforms.Compose([
            transforms.RandomResizedCrop(model.module.visual.input_resolution 
                                         if (isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel)) else
                                         model.visual.input_resolution),
            transforms.RandomHorizontalFlip(),
            preprocess,
        ]))

    val_dataset = CocoDetectionCaption(
        valdir,
        valjson,
        valcapjson,
        preprocess)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for param in model.parameters():
        param.requires_grad = False

    if (isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel)):
        for param in model.module.image_text.parameters():
            param.requires_grad = True 
    else:
        for param in model.image_text.parameters():
                param.requires_grad = True 

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    num_var = sum(p.numel() for p in params)

    print('Training %d variables' % num_var)

    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)

            #print(checkpoint['state_dict'])

            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        
        scheduler.step()

        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        #print(target.min().data.cpu(), target.max().data.cpu()) 

        # compute output
        output = model(images, text_tokens)

        print(output)
        print(target)

        loss = criterion(output, target)

        # measure accuracy_multilabel and record loss
        acc1, acc5 = accuracy_multilabel(output, target, threshs=(.2, .5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)

def validate(val_loader, model, criterion, args):

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses1 = AverageMeter('Loss 1', ':.4e', Summary.NONE)
    losses2 = AverageMeter('Loss 2', ':.4e', Summary.NONE)
    losses3 = AverageMeter('Loss 3', ':.4e', Summary.NONE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses1, losses2, losses3],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    base_progress = 0
    with torch.no_grad():
        end = time.time()
        for i, (images, classes_prompt_tokens, caption_tokens) in enumerate(val_loader):               

            i = base_progress + i

            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)
                classes_prompt_tokens = classes_prompt_tokens.cuda(args.gpu, non_blocking=True)
                caption_tokens = caption_tokens.cuda(args.gpu, non_blocking=True)

            # compute output
            image_features, classes_prompt_features, caption_features = model(images, classes_prompt_tokens, caption_tokens)

            loss1 = 1-criterion(image_features, classes_prompt_features)
            loss2 = 1-criterion(classes_prompt_features, caption_features)
            loss3 = 1-criterion(image_features, caption_features)

            # measure accuracy_multilabel and record loss
            losses1.update(loss1.item(), images.size(0))
            losses2.update(loss2.item(), images.size(0))
            losses3.update(loss3.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)

    progress.display_summary()

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):

    if "AMLT_OUTPUT_DIR" in os.environ:
        filename = os.path.join(os.environ["AMLT_OUTPUT_DIR"], filename)

    print('saving ' + filename)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        total = torch.FloatTensor([self.sum, self.count])
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy_multilabel(output, target, threshs=(.5,.9)):
    with torch.no_grad():
        outputs = torch.sigmoid(output)

        accs = []

        for thresh in threshs:
            outputb = torch.ge(outputs, thresh)
            targetb = target.bool()

            tp = torch.sum(torch.bitwise_and(outputb, targetb).int()).cpu().detach().numpy()            
            #tn = torch.sum(torch.logical_not(torch.bitwise_or(outputb, targetb)).int()).cpu().detach().numpy()

            total = targetb.size(1)

            acc = tp/total
            accs.append(acc)

        return accs


if __name__ == '__main__':
    main()
