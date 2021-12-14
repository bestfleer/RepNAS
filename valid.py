import os
import random
from cnn import utils
import logging
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from torchvision.transforms import transforms
import torchvision.datasets as datasets

parser = argparse.ArgumentParser("RepNAS valid")
parser.add_argument('--data', type=str, default='.', help='location of the data')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--model', type=str, default='RepVGGA0', help='type of model which can be selected in [RepVGG_A0, RepVGG_A1, RepVGG_B2g4, RepVGG_B3]')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--pretrained', type=str, default='./logs/a0.pt', help='location of the best model')
parser.add_argument('--mode', type=str, default='supernet', help='fuse supernet or subnet')
parser.add_argument('--seed', type=int, default=2, help='random seed')


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
  cudnn.benchmark = True
  cudnn.enabled = True
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

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
  criterion = torch.nn.CrossEntropyLoss().cuda()
  if args.mode == 'supernet':
    from cnn.supernet import model_map
    model = model_map[args.model]()
    print('loading pretrained model ...')
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.fixed_path = checkpoint['fixed_path']
    print(model.fixed_path)
  elif args.mode == 'subnet':
    from cnn.subnet import model_map
    from arch import model_arch
    model = model_map[args.model](model_arch[args.model])
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    print(model_arch[args.model])
  print('fusing ...')
  model.fuse_weights()
  model = model.cuda()
  print('validating ...')
  acc, loss = infer(valid_queue, model, criterion)
  print(acc, loss)


def infer(valid_queue, model, criterion):
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
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == '__main__':
  main()