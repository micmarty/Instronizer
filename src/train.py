
# Installed modules
from pathlib import Path
import argparse
import shutil
import time
import torch
import torch.utils.data
import better_exceptions

# Custom utils
from models.mobilenet import MobileNet
from logger import Logger # Tensorboard
from utils.average_meter import AverageMeter
from utils import printing_functions as pf
from utils import dataset_finder as df

def input_args():
    parser = argparse.ArgumentParser('PyTorch Instrument classifier training script')

    # Important
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--gpu', dest='use_cuda', action='store_true', help='Use CUDA')

    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default for mobilenet: 256)')
    parser.add_argument('-l', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate (should be adjusted to batch size, default for mobilenet: 0.1)')
    parser.add_argument('-n', '--num-classes', default=11, type=int, help='Number of classes for input dataset (default: 11 -> IRMAS dataset')
    parser.add_argument('-I', '--image-size', default=224, type=int, help='Input images size (must be square, default: 224')

    # Epoch related
    parser.add_argument('-s', '--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-e', '--epochs', default=90, type=int, metavar='N', help='number of epochs to run (default: 90), counting from the start-epoch (default: 0)')

    # Runtime related
    parser.add_argument('-R', '--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-P', '--print-freq',  default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-T', '--tensorboard-freq', default=10, type=int, help='Updating tensorboard state in iterations (default: 10)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    # Other
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-d', '--weight-decay',  default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--top-k',  default=3, type=int, metavar='K', help='Tolerance for K model outputs with highest activations')

    # Unused
    # parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    # parser.add_argument('--test-spectrograms', metavar='PATH', type=str, help='Directory with spectrograms to classify')
    return parser.parse_args()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_checkpoint(model, optimizer):
    '''
    Loads a single file with path specified inside args.
    It restores both: model and optimizer state.
    '''
    global args
    if Path(args.resume).is_file():
        print("[load_checkpoint]: Loading checkpoint from '{}'".format(args.resume))

        # Retrieve persisted data
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['start_epoch']
        best_precision_1 = checkpoint['best_precision_1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('[load_checkpoint]: Checkpoint loaded successfully!')
    else:
        print("[load_checkpoint]: No checkpoint found at '{}'".format(args.resume))

def load_data_from_folder(folder):
    '''
    Returns a data loader object which:
    - contains all spectrogram files found under <folder> path, paired with labels,
    - groups the data into batches
    - shuffles the data, etc.
    '''
    
    path = Path(args.data, folder)
    if folder == 'train':
        dataset = df.SpecFolder(str(path))
    elif folder == 'val':
        dataset = df.ValSpecFolder(str(path))
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.workers)

def adjust_learning_rate(optimizer, epoch):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def print_training_step(epoch, step, data_size, batch_time, data_time, losses, top1, topk):
    if step % args.print_freq == 0:
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@{3} {topk.val:.3f} ({topk.avg:.3f})'.format(
                  epoch, step, data_size, args.top_k,
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  top1=top1,
                  topk=topk))

def print_validation_step(step, data_size, batch_time, losses, top1, topk):
    if step % args.print_freq == 0:
        print('Validation: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@{2} {topk.val:.3f} ({topk.avg:.3f})'.format(
                  step, data_size, args.top_k,
                  batch_time=batch_time,
                  loss=losses,
                  top1=top1,
                  topk=topk))

def to_np(value):
    # TODO make sure it will work on GPU
    return value.data.cpu().numpy()

def log_to_tensorboard(model, step, input_var, losses, top1, topk, mode):
    im = {'amount': 10, 'every': 3}
    
    if step % args.tensorboard_freq == 0:
        # (1) Log the scalar values
        info = {
            '{}_loss_avg'.format(mode): losses.avg,
            '{}_loss'.format(mode): losses.val,

            '{}_accr_avg'.format(mode): top1.avg,
            '{}_accr'.format(mode): top1.val,

            '{}_accr_top{}_avg'.format(mode, args.top_k): topk.avg,
            '{}_accr_top{}'.format(mode, args.top_k): topk.val
        }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)

        # (2) Log values and gradients of the parameters (histogram)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, to_np(value), step)

            # At the beginning it may happen
            # source: https://discuss.pytorch.org/t/zero-grad-optimizer-or-net/1887/6
            if value.grad is not None:
                logger.histo_summary(tag + '/grad', to_np(value.grad), step)


        # (3) Log the images
        # Take non-repeating images (every=3) in specified amount
        images = input_var.view(-1, args.image_size, args.image_size)[:(im['amount'] * im['every'])]
        images = images[::im['every']]
        info = {
            '{}_images'.format(mode): to_np(images)
        }
        for tag, images in info.items():
            logger.image_summary('{}_{}'.format(mode, tag), images, step)

def accuracy(output, target, topk=(1, 3)):
    '''Computes the precision@k for the specified values of k'''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(training_data, model, criterion, optimizer, epoch):
    '''Executes 1 epoch training'''
    batch_time, data_time, losses, top1, topk = [AverageMeter()] * 5
    model.train()

    timer_start = time.time()
    for step, (input, target) in enumerate(training_data):
        # Measure data loading performance
        data_time.update(time.time() - timer_start)

        if args.use_cuda:
            input = target.cuda(async=True)
            target = target.cuda(async=True) 
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # We need to reset the gradient
        # source: https: // discuss.pytorch.org /t/zero-grad-optimizer-or-net/1887/3
        optimizer.zero_grad()

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # Propagate changes
        loss.backward()
        optimizer.step()

        # Measure accuracy and record loss
        # prec_1 defines how close the output was to the target
        # prec_k defines the same but is tolerant for K instruments with highest activation
        #
        # Example: target is cello, but the output was: 
        # (from best to worse) guitar, voice, cello, other instruments ...
        # if top_k was set to 3 we treat such an output as 'good enough' because cello one of those top3 responses
        prec_1, prec_k = accuracy(output.data, target, topk=(1, args.top_k))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec_1[0], input.size(0))
        topk.update(prec_k[0], input.size(0))

        # Measure batch performance and update timer
        batch_time.update(time.time() - timer_start)
        timer_start = time.time()

        # Print to the console and log to Tenorboard
        print_training_step(epoch, step, len(training_data), batch_time, data_time, losses, top1, topk)
        log_to_tensorboard(model, step, input_var, losses, top1, topk, mode='train')

def string_to_class_idx(strings):
    '''Transforms a list of strings into 2d list
    Array element: '1;2;3;5;9;'
    Returned element: [1,2,3,5,9]

    Array: ['1;2;3;5;9;', '1;2;', '1;2;9;']
    Returns: [[1,2,3,5,9], [1,2], [1,2,9]]
    '''
    result = []
    for label_string in strings:
        # Remove trailing semicolon and split by this separator
        labels = label_string[:-1].split(';')
        # Convert all elements from string to int
        result.append(list(map(int, labels)))
    return result

def onehot(y):
    y_onehot = torch.FloatTensor(args.num_classes).zero_()
    for i in y:
        y_onehot[i] = 1
    return y_onehot

def onehot_2d(classes_list):
    result = torch.FloatTensor(args.batch_size, args.num_classes)
    for row, classes in enumerate(classes_list):
        classes = torch.LongTensor(classes)
        result[row, :] = onehot(classes)
    return result
        
def validate(validation_data, model, criterion):
    batch_time, losses, top1, topk = [AverageMeter()] * 4
    model.eval()

    timer_start = time.time()
    for step, (input, target) in enumerate(validation_data):
        
        target = onehot_2d(string_to_class_idx(target))
        if args.use_cuda:
            input = target.cuda(async=True)
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        '''TODO fix and uncomment code below'''
        # prec_1, prec_k = accuracy(output.data, target, topk=(1, args.top_k))
        losses.update(loss.data[0], input.size(0))
        # top1.update(prec_1[0], input.size(0))
        # topk.update(prec_k[0], input.size(0))

        # Measure batch performance and update timer
        batch_time.update(time.time() - timer_start)
        timer_start = time.time()

        # Print to the console and log to Tenorboard
        print_validation_step(step, len(validation_data), batch_time, losses, top1, topk)
        log_to_tensorboard(model, step, input_var, losses, top1, topk, mode='val')
    print(' * Prec@1 {top1.avg:.3f} Prec@{} {topk.avg:.3f}'.format(args.top_k, top1=top1, topk=topk))
    return top1.avg


def run_training(training_data, validation_data, model, criterion, val_criterion, optimizer):
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(training_data, model, criterion, optimizer, epoch)
        precision_1 = validate(validation_data, model, val_criterion)

        best_precision_1 = max(precision_1, best_precision_1)
        is_best = precision_1 > best_precision_1

        save_checkpoint({
            'model': model.__class__.__name__,
            'arch': model.__class__.__name__,
            'start_epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_precision_1': best_precision_1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

def main():
    global args, logger, best_precision_1
    args = input_args()
    logger = Logger('./logs')   # Tensorboard logger
    best_precision_1 = 0          # Global record

    pf.print_args(args) 

    # IMPORTANT model .cuda() needs to be called before optimizer definition!
    # source: http://pytorch.org/docs/master/optim.html
    model = MobileNet(num_classes=args.num_classes)
    if args.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
    print(model)

    # TODO consider Adam as an optimizer function
    criterion = torch.nn.CrossEntropyLoss()
    val_criterion = torch.nn.MultiLabelSoftMarginLoss()
    if args.use_cuda:
        criterion = criterion.cuda()
        val_criterion = val_criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.learning_rate, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    
    if args.resume:
        load_checkpoint(model, optimizer)
    
    if args.evaluate:
        validation_data = load_data_from_folder('val')
        validate(validation_data, model, val_criterion)
    else:
        training_data = load_data_from_folder('train')
        validation_data = load_data_from_folder('val')
        run_training(training_data, validation_data, model, criterion, val_criterion, optimizer)

if __name__ == '__main__':
    main()
