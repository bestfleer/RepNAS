import sys
import random
import time
from cnn import utils
import logging
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from utils.scheduler import Scheduler
from torchvision.transforms import transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter
from cnn.supernet import *
from distributed import *
from utils.auto_augment import auto_augment_transform
from utils.Mixup import Mixup
from utils.loss import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser("RepNAS")
parser.add_argument('--data', type=str, default='.', help='location of the data')
parser.add_argument('--workers', type=int, default=50, help='number of data loading workers')
parser.add_argument('--model', type=str, default='RepVGGA0', help='type of model which can be selected in [RepVGG_A0, RepVGG_A1, RepVGG_B2g4, RepVGG_B3]')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--base_lr', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--lr_mode', type=str, default='cosine', help='[step, poly, cosine]')
parser.add_argument('--wd_mode', type=str, default='cosine', help='[step, poly, cosine]')
parser.add_argument('--warmup_lr', type=float, default=0.1, help='init warmup learning rate')
parser.add_argument('--warmup_epochs', type=int, default=0, help='number of warmup epochs')
parser.add_argument('--warmup_mode', type=str, default='constant', help='mode of warmup [constant, linear]')
parser.add_argument('--mixup', action='store_true', help='using mixup')
parser.add_argument('--autoaugment', action='store_true', help='using autoaugment')
parser.add_argument('--smooth', action='store_true', help='using smooth CE')
parser.add_argument('--report_freq', type=float, default=500, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoint.pt', help='save checkpoint')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--save', type=str, default='logs', help='experiment name')
parser.add_argument('--random_epochs', type=int, default=15, help='number of random sample epochs')
parser.add_argument('--fixed_epochs', type=int, default=0, help='number of fixed epochs.')
parser.add_argument('--arch_lr', type=float, default=1e-3, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-4, help='weight decay for arch encoding')
parser.add_argument('--local_rank', type=int, default=0, help='number for current rank')



args = parser.parse_args()

CLASSES = 1000

def set_random_seed(seed=None):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
  args.distributed = False
  if 'WORLD_SIZE' in os.environ:
      args.distributed = int(os.environ['WORLD_SIZE']) > 1
      args.batch_size = args.batch_size // int(os.environ['WORLD_SIZE'])
  if args.distributed:
    gpu_id = init_dist()
    set_random_seed(args.seed)
  if is_master():
      Writer = SummaryWriter(log_dir=current_time)
      print(args)
  else:
      Writer = None
  cudnn.benchmark = True
  cudnn.enabled = True
  if not args.mixup and args.smooth:
      criterion_smooth = LabelSmoothingCrossEntropy().cuda()
  elif args.mixup and args.smooth:
      criterion_smooth = SoftTargetCrossEntropy().cuda()
  else:
      criterion_smooth = nn.CrossEntropyLoss().cuda()
  criterion = nn.CrossEntropyLoss().cuda()
  model = model_map[args.model]()
  model = model.cuda()
  param = []
  arch_param = []
  for key, value in model.named_parameters():
      if not value.requires_grad:
          continue
      if 'alphas' in key:
          arch_param += [{'params': [value], 'lr': args.arch_lr, 'weight_decay': args.arch_weight_decay}]
      else:
          if 'bias' in key or 'bn' in key:
              weight_decay = 0
          else:
              weight_decay = args.weight_decay
          param += [{'params': [value], 'lr': args.base_lr, 'weight_decay': weight_decay}]
  optimizer = torch.optim.SGD(
      param,
      momentum=args.momentum,
      )
  arch_optimizer = torch.optim.Adam(
      arch_param,
      betas=(0.5, 0.999),
      weight_decay=args.arch_weight_decay
  )
  current_epoch = 0
  if os.path.exists(os.path.join(args.save, 'ckpt.pt')):
      print('loading checkpoint')
      checkpoint = torch.load(os.path.join(args.save, 'ckpt.pt'), map_location='cpu')
      current_epoch = checkpoint['epoch']
      state_dict = OrderedDict()
      for name, param in checkpoint['model'].items():
        if 'alpha' not in name:
            state_dict[name] = param
      model.load_state_dict(state_dict, strict=False)
      optimizer.load_state_dict(checkpoint['optimizer'])
      arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
      model.fixed_path = checkpoint['fixed_path']
  if is_master():
    print("total path = {}".format(model.original_ops))
  model.constraint = 0.34
  if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)
    model_without_ddp = model.module
  else:
    model_without_ddp = model
  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  if args.autoaugment:
      transformer = transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          auto_augment_transform('original', dict(translate_const=int(224*0.45), img_mean=tuple([min(255, round(255 * x)) for x in [0.485, 0.456, 0.406]]))),
          transforms.ToTensor(),
          normalize,
      ])
  else:
      transformer = transforms.Compose([
          transforms.RandomResizedCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize,
      ])
  train_dataset = datasets.ImageFolder(
      traindir,
      transformer)

  if args.distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  else:
      train_sampler = None

  train_queue = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
      num_workers=args.workers, pin_memory=True, sampler=train_sampler)

  val_dataset = datasets.ImageFolder(validdir, transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
  ]))
  valid_queue = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=args.batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True)
  mixup_fn = Mixup() if args.mixup else None
  if is_master():
      print("step num for each epoch:", len(train_queue))
  lr_scheduler = Scheduler(optimizer, len(train_queue), 'lr', args.epochs, base_value=args.base_lr)
  lr_scheduler.update(0, current_epoch)
  wd_scheduler = Scheduler(arch_optimizer, len(train_queue),
                           'weight_decay', args.epochs - args.random_epochs - args.fixed_epochs,
                           base_value=args.arch_weight_decay)
  wd_scheduler.update(0, max(0, current_epoch - args.random_epochs))
  best_top1 = 0
  for i in range(current_epoch, args.epochs):
      if args.distributed:
        train_queue.sampler.set_epoch(i)
      if is_master():
        print('epoch {}'.format(i))
      if i < args.random_epochs:
          model_without_ddp.random_sample = True
      elif i == args.random_epochs and is_master():
          torch.save({'model': model_without_ddp.state_dict(),
                      'epoch': i,
                      'optimizer': optimizer.state_dict(),
                      'arch_optimizer': arch_optimizer.state_dict(),
                      'fixed_path': model_without_ddp.fixed_path
                      }, os.path.join(args.save, 'pretrained.pt'))
      else:
          model_without_ddp.random_sample = False
      if i >= args.epochs - args.fixed_epochs and model_without_ddp.fixed_path is None:
          model_without_ddp.fixed_mask()
          if is_master():
              print("fixed path:", model_without_ddp.fixed_path)
      train(args, train_queue, model, model_without_ddp, optimizer, arch_optimizer, lr_scheduler, wd_scheduler, i, mixup_fn, criterion_smooth, Writer)
      top1 = infer(valid_queue, model, criterion, i, Writer)
      if is_master():
          print(model_without_ddp.alphas.sigmoid())
          print("epoch:{} top1:{:3f}".format(i, top1))
          torch.save({'model': model_without_ddp.state_dict(),
                      'epoch': i+1,
                      'optimizer': optimizer.state_dict(),
                      'arch_optimizer': arch_optimizer.state_dict(),
                      'fixed_path': model_without_ddp.fixed_path
                      }, os.path.join(args.save, 'ckpt.pt'))
          if top1 > best_top1:
              torch.save({'model': model_without_ddp.state_dict(),
                          'epoch': i + 1,
                          'top1': top1,
                          'fixed_path': model_without_ddp.fixed_path}, os.path.join(args.save, 'best.pt'))
              best_top1 = top1
  if is_master():
      Writer.close()

def train(args, train_queue, model, model_without_ddp, optimizer, arch_optimizer, lr_scheduler, wd_scheduler, epoch, mixup_fn, criterion, Writer):
    obj = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    if mixup_fn is not None:
        mixup_fn.mixup_enabled = False
    for step, (inputs, targets) in enumerate(train_queue):
        optimizer.zero_grad()
        arch_optimizer.zero_grad()
        inputs = inputs.cuda()
        targets = targets.cuda()
        if mixup_fn is not None:
            inputs, smooth_targets = mixup_fn(inputs, targets)
        else:
            inputs, smooth_targets = inputs, targets
        logits = model(inputs)
        loss = criterion(logits, smooth_targets)
        loss.backward()
        nn.utils.clip_grad_norm(model_without_ddp.parameters(), 5)
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        if args.distributed:
            dist_all_reduce_tensor(loss)
            dist_all_reduce_tensor(prec1)
            dist_all_reduce_tensor(prec5)
        obj.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        optimizer.step()
        if not model_without_ddp.random_sample and model_without_ddp.fixed_path is None:
            arch_optimizer.step()
            wd_scheduler.update(step, epoch - args.random_epochs)
        lr_scheduler.update(step, epoch)
        if any(torch.isnan(model_without_ddp.alphas.view(-1))):
            sys.exit(1)
        if step % args.report_freq == 0 and is_master():
            # print(model.alphas.sigmoid())
            print('train step:{}\
             lr:{:.3f} wd:{:.3f}\
              loss:{:.3f} top1:{:.3f}\
               top5:{:.3f}'.format(step,
                                   lr_scheduler.value,
                                   wd_scheduler.value,
                                   obj.avg, top1.avg, top5.avg))
            Writer.add_scalar('train loss', obj.avg, epoch*len(train_queue)+step)
            Writer.add_scalar('train top1 acc', top1.avg, epoch*len(train_queue)+step)

def infer(valid_queue, model, criterion, epoch, Writer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda()
      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      dist_all_reduce_tensor(loss)
      dist_all_reduce_tensor(prec1)
      dist_all_reduce_tensor(prec5)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

  if is_master():
      Writer.add_scalar('valid loss', objs.avg, epoch)
      Writer.add_scalar('valid top1 acc', top1.avg, epoch)
  return top1.avg

if __name__ == '__main__':
  main()