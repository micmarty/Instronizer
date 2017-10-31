# TODO optimize imports
# TODO clean up this mess, too much in one file!

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
import torchvision.models
import better_exceptions
import numpy as np
from logger import Logger

from models.mobilenet import MobileNet
from utils import printing_functions as pf
import dataset_finder as df

# Prepare set of available model names (callable from the command-line)
model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))
model_names.append('mobilenet')


# Parser - all default settings are stored here
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
                    help='number of epochs to run (default: 4), counting from the start-epoch (default: 0)')

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

parser.add_argument('--num-classes', default=11, type=int,
                    help='Number of classes for input dataset (default: 11')

parser.add_argument('--test-spectrograms', metavar='PATH', type=str,
                    help='Directory with spectrograms to classify')

def main():
    # Make them available for all functions
    global args, best_prec1, logger, classes

    args = parser.parse_args()
    pf.print_args(args)

    # Best Precision Measure
    best_prec1 = 0

    # Tensorboard logger
    logger = Logger('./logs')

    # Default classes
    classes = ['x'] * args.num_classes

    # Create model (torchvision or custom/project-defined)
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = torchvision.models.__dict__[args.arch](
            pretrained=args.pretrained, num_classes=args.num_classes)
    else:
        print("=> creating model '{}'".format(args.arch))
        if args.arch.startswith('mobilenet'):
            model = MobileNet(num_classes=args.num_classes)
        else:
            model = torchvision.models.__dict__[
                args.arch](num_classes=args.num_classes)
    print('=== MODEL ARCHITECTURE ======================')
    print(model)
    print('=== END =====================================\n')

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            # Retrieve stored data
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Dataset paths
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # TODO calculate np.mean and std on whole dataset before training manually (@moonman)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    # Train + Validation sets
    if not args.test_spectrograms:
        train_dataset = df.SpecFolder(traindir)

        classes = train_dataset.classes
        train_loader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers)

        val_loader = torch.utils.data.DataLoader(
            df.SpecFolder(traindir),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers)
    else:
        # TODO ImageFolder requires to have label folder, like: train/cello, train/piano
        # But when we want to run script in test-only mode then we HAVE TO create fake folder <spectrograms_to_classify_path>/fake/<actual_files> 
        testdir = args.test_spectrograms
        testdataset = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Scale(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ]))
        print('{} testing files found.'.format(len(testdataset)))

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
            filename='checkpoint__{}__start_epoch_{}__best_prec_{0:.4f}.pth.tar'.format(model.__class__.__name__, epoch + 1, best_prec1))



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
        prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
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
    
def validate(val_loader, model, criterion):
    print('=== VALIDATING ====================')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    predictions_timeline = []
    predictions_counter = [0] * args.num_classes
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # TODO use label names in tensorboard logging (images tab)
        pf.print_validation_info(target, output.data, 
                                class_names=classes, 
                                classes_on_timeline=predictions_timeline, 
                                classes_counter=predictions_counter)
        #print("[DEBUG] Target: \n{}".format(list(target))
        #print("[DEBUG] Predicted output:\n{}".format(output))
        #print("[DEBUG] Predicted label: {}".format(class_names[]))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 3))
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
    predictions_timeline = []
    predictions_counter = [0] * len(classes)
    for i, (input, target) in enumerate(test_loader):
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        output = model(input_var)
        pf.print_test_info(output.data, class_names=classes,
                           classes_on_timeline=predictions_timeline,
                           classes_counter=predictions_counter, current_index=i, max_index=len(test_loader))

    pf.print_class_counters(classes, predictions_counter)
    # Print prediction timeline
    print('\nPredictions timeline:')
    for instrument_index in predictions_timeline:
        print(instrument_index, end='') 
    print('\n=== END ====================')


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
