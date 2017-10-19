import argparse
import os
import os.path
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import better_exceptions
import numpy as np
from logger import Logger

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

model_names.append('mobilenet')


''' 1. Parser '''
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=4, type=int, metavar='N',
                    help='number of epochs to run (default: 4)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=6, type=int,
                    metavar='N', help='mini-batch size (lowered to: 6, default for mobilenet: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.0024, type=float,
                    metavar='LR', help='initial learning rate (lowered beacuse of batch size to: 0.0024, default for mobilenet: 0.1)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')

parser.add_argument('--tensorboard-freq', default=10, type=int,
                    help='Updating tensorboard state in iterations (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--classify-spectrograms', dest='classify', action='store_true',
                    help='evaluate model on validation set without knowledge about the labels')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

parser.add_argument('--image-size', default=224, type=int,
                    help='Input images size (must be square, default: 224')

parser.add_argument('--num-classes', default=3, type=int,
                    help='Number of classes for input dataset (default: 3')

parser.add_argument('--test-spectrograms', metavar='PATH', type=str,
                    help='Directory with spectrograms to classify')

# Model
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3,  32, 2),
            conv_dw(32,  64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, args.num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def main():
    global args, best_prec1, logger
    args = parser.parse_args()
    best_prec1 = 0
    logger = Logger('./logs')
    
    print_args()

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('mobilenet'):
            model = MobileNet()
            print('=== MODEL ARCHITECTURE ======================')
            print(model)
            print('=== END =====================================\n')
        else:
            model = models.__dict__[args.arch]()


    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # TODO calculate np.mean and std on whole dataset before training manually (@moonman)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if not args.test_spectrograms:
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, transforms.Compose([
                # transforms.RandomSizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Scale(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Scale(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers)
    else:
        # TODO fix that -> it's a bad practice (ImageFolder requires to have label folder, so I ommited it...doing so)
        #testdir = os.path.abspath(os.path.join(args.test_spectrograms, os.pardir))
        testdir = args.test_spectrograms

        print(testdir)
        testdataset = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Scale(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ]))
        print(len(testdataset))
        test_loader = torch.utils.data.DataLoader(testdataset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

        test(test_loader, model, criterion)
        return

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'model': model.__class__.__name__,
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, 
        filename='checkpoint__{}__start_epoch_{}__best_prec_{}.pth.tar'.format(model.__class__.__name__, epoch+1, best_prec1))

def print_args():
    print('=== PARAMETERS ==============================')
    for arg in vars(args):
        print(arg.upper(), '=', getattr(args, arg))
    print('=== END =====================================\n\n')

def to_np(x):
    return x.data.cpu().numpy()


def train(train_loader, model, criterion, optimizer, epoch):
    print('=== TRAINING ====================')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    # Save for plotting
    loss_vals = []
    loss_avgs = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, args.num_classes))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        topk.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                  'Prec@k {topk.val:.3f} ({topk.avg:.3f})'.format(
                   epoch, i, len(train_loader),
                   batch_time=batch_time, 
                   data_time=data_time, 
                   loss=losses, 
                   top1=top1, 
                   topk=topk))

        # TODO use arg parameter for that
        if i % 200 == 0:
            save_checkpoint({
                'model': model.__class__.__name__,
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best=False,
                filename='checkpoint__{}__curr_epoch_{}__iter_{}.pth.tar'.format(model.__class__.__name__, epoch, i))

        if i % args.tensorboard_freq == 0:
            #============ TensorBoard logging ============#
            # (1) Log the scalar values
            info={
                'loss_avg': losses.avg,
                'loss': losses.val,
                'accuracy_avg': top1.avg,
                'accuracy': top1.val
            }
            #print(losses.avg, losses.val, top1.val)
            for tag, value in info.items():
                logger.scalar_summary(tag, value, i)

            # (2) Log values and gradients of the parameters (histogram)
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, to_np(value), i)
                logger.histo_summary(tag + '/grad', to_np(value.grad), i)

            # (3) Log the images
            info = {
                'train_images': to_np(input_var.view(-1, args.image_size, args.image_size)[:10])
            }
            for tag, images in info.items():
                logger.image_summary('{}_{}'.format('train_debug_prefix', tag), images, i)
    print('=== END ====================')
    
# MUST BE ALPHABETICAL ORDER
# TODO move somewhere else! 
def print_validation_info(target, output_data, result=None):
    class_names = ['cello', 'piano', 'ukulele']
    predictions = [np.argmax(output_row.numpy()) for output_row in output_data]
    pairs = zip(list(target), predictions)

    print("Model output: {}".format(output_data))
    print('Target     | Prediction')
    print("-----------------------")
    for pair in pairs:
        if result:
            result[pair[1]] += 1
        print('{:10s} | {:10s}'.format(class_names[pair[0]], class_names[pair[1]]))
    print()


def validate(val_loader, model, criterion):
    print('=== VALIDATING ====================')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # TODO use label names in tensorboard logging (images tab)
        print_validation_info(target, output.data)
        #print("[DEBUG] Target: \n{}".format(list(target))
        #print("[DEBUG] Predicted output:\n{}".format(output))
        #print("[DEBUG] Predicted label: {}".format(class_names[]))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, args.num_classes))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        topk.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {topk.val:.3f} ({topk.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, topk=topk))
            #============ TensorBoard logging ============#
            # It is called with print_freq, not tensorboard_freq!
            # because validation set is smaller
            # (1) Log the scalar values
            info = {
                'validation_loss_avg': losses.avg,
                'validation_loss': losses.val,
                'validation_accuracy_avg': top1.avg,
                'validation_accuracy': top1.val
            }
            print(losses.avg, losses.val, top1.val)
            for tag, value in info.items():
                logger.scalar_summary(tag, value, i)

            # (3) Log the images
            info = {
                'val_images': to_np(input_var.view(-1, args.image_size, args.image_size)[:10])
            }
            for tag, images in info.items():
                logger.image_summary('{}_{}'.format('val_debug_prefix', tag), images, i)

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {topk.avg:.3f}'
          .format(top1=top1, topk=topk))
    print('=== END ====================')
    return top1.avg


def test(test_loader, model, criterion):
    print('=== TESTING ====================')

    # switch to evaluate mode
    model.eval()
    result = [0,0,0]
    for i, (input, target) in enumerate(test_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        print_validation_info(target, output.data, result=result)
    print('Cello: {} | Piano: {} | Ukulele: {}'.format(result[0], result[1], result[2]))
    print('=== END ====================')


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print("Label: {}, Prediction: {}".format(
    #    target, pred[0, :]))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
