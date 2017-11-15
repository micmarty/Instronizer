
# Installed modules
from pathlib import Path
import argparse
import shutil
import time
import torch
import torch.utils.data
import better_exceptions

# Relative to application application source root
from classifier.models.mobilenet import MobileNet
from classifier.models.densenet import densenet161
from classifier.utils.tensorboard_logger import Logger  # Tensorboard
from classifier.utils.average_meter import AverageMeter
from classifier.utils import printing_functions as pf
import classifier.dataset_loader as df

def input_args():
    parser = argparse.ArgumentParser('PyTorch Instrument classifier training script')

    # Important
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--gpu', dest='use_cuda', action='store_true', help='Use CUDA')

    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default for mobilenet: 256)')
    parser.add_argument('--val-batch-size', default=150, type=int, metavar='N', help='mini-batch size for validation dataset')
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


def save_checkpoint(state, is_best, start_epoch, filename='checkpoint.pth.tar'):
    # Store first n checkpoints in a single file (overriding)
    if start_epoch < 10:
        torch.save(state, filename)
    else:
        filename = 'checkpoint_{}.pth.tar'
        torch.save(state, filename.format(start_epoch))
    if is_best:
        shutil.copyfile(filename, 'model_best_{}.pth.tar'.format(start_epoch))

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
        batch_size = args.batch_size
    elif folder == 'val':
        dataset = df.SpecFolder(str(path))
        batch_size = args.val_batch_size
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=args.workers)

def adjust_learning_rate(optimizer, epoch):
    '''Sets the learning rate to the initial LR decayed by 10 every 30 epochs'''
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    print("Current learning rate: {}".format(lr))
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
    
    if step % args.tensorboard_freq == 0 or mode == 'val':
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
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()

    model.train()

    timer_start = time.time()
    for step, (input, target) in enumerate(training_data):
        # Measure data loading performance
        data_time.update(time.time() - timer_start)

        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda() 
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)


        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

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

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure batch performance and update timer
        batch_time.update(time.time() - timer_start)
        timer_start = time.time()

        # Print to the console and log to Tenorboard
        print_training_step(epoch, step, len(training_data), batch_time, data_time, losses, top1, topk)
        log_to_tensorboard(model, step, input_var, losses, top1, topk, mode='train')



def validate_single_labeled(validation_data, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    model.eval()

    timer_start = time.time()
    for step, (input, target) in enumerate(validation_data):
        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        # Volatile option means: "Don't calculate the gradients"
        # We don't need gradients because no backward() functions is called
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # Compute the output
        output = model(input_var)
        loss = criterion(output, target_var)

        prec_1, prec_k = accuracy(output.data, target, topk=(1, args.top_k))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec_1[0], input.size(0))
        topk.update(prec_k[0], input.size(0))

        # Measure batch performance and update timer
        batch_time.update(time.time() - timer_start)
        timer_start = time.time()

        # Print to the console and log to Tenorboard
        print_validation_step(step, len(validation_data),
                              batch_time, losses, top1, topk)

        # We don't need to disort plot with chaotic points
        # Log to tensorboard only once per validation phase (see run_training function)
        log_to_tensorboard(model, step, input_var, losses, top1, topk, mode='val')
    print(
        ' * Prec@1 {top1.avg:.3f} Prec@{} {topk.avg:.3f}'.format(args.top_k, top1=top1, topk=topk))
    return top1.avg


def run_training(training_data, validation_data, model, criterion, optimizer):
    global best_precision_1

    # Validate before training (how good model is guessing on random weight)
    precision_1 = validate_single_labeled(validation_data, model, criterion)
    logger.scalar_summary('validation_overall', precision_1, 0)

    # Run normal train-validate cycle
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(training_data, model, criterion, optimizer, epoch)
        precision_1 = validate_single_labeled(validation_data, model, criterion)
        
        # Log overall validation performance on separate chart
        logger.scalar_summary('validation_overall', precision_1, epoch)

        best_precision_1 = max(precision_1, best_precision_1)
        is_best = precision_1 > best_precision_1

        save_checkpoint({
            'model': model.__class__.__name__,
            'arch': model.__class__.__name__,
            'start_epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_precision_1': best_precision_1,
            'optimizer': optimizer.state_dict(),
        }, is_best, start_epoch=epoch + 1)


def main():
    global args, logger, best_precision_1
    args = input_args()
    logger = Logger('./tensorboard_logs')   # Tensorboard logger
    best_precision_1 = 0          # Global record

    pf.print_args(args) 

    # IMPORTANT model.cuda() needs to be called before optimizer definition!
    # source: http://pytorch.org/docs/master/optim.html
    model = MobileNet(num_classes=args.num_classes)
    #model = densenet161(drop_rate=0.2, num_classes=6)
    if args.use_cuda:
        # Removed DataParallel because that was the reason of
        # failures when loading checkpoints onto CPU
        model.cuda()
    print(model)

    # TODO consider Adam as an optimizer function
    criterion = torch.nn.CrossEntropyLoss()

    if args.use_cuda:
        criterion = criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.learning_rate, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)
    
    if args.resume:
        load_checkpoint(model, optimizer)
    
    if args.evaluate:
        validation_data = load_data_from_folder('val')
        validate_single_labeled(validation_data, model, criterion)
    else:
        validation_data = load_data_from_folder('val')
        training_data = load_data_from_folder('train')
        run_training(training_data, validation_data, model, criterion, optimizer)

if __name__ == '__main__':
    main()


# UNUSED CODEBASE for multilabel validation on IRMAS
#
# def run_validation(model, criterion):
#     root = Path(args.data, 'val')
#     spec_folder = df.ValSpecFolder(str(root))

#     for val_song_idx in range(20):
#         validation_data = torch.utils.data.DataLoader(spec_folder,
#                                                   batch_size=args.val_batch_size,
#                                                   shuffle=False,
#                                                   num_workers=args.workers)
#         validate(validation_data, model, criterion)
#         validation_data.dataset.next_song()

# def string_to_class_idx(strings):
#     '''Transforms a list of strings into 2d list
#     Array element: '1;2;3;5;9;'
#     Returned element: [1,2,3,5,9]

#     Array: ['1;2;3;5;9;', '1;2;', '1;2;9;']
#     Returns: [[1,2,3,5,9], [1,2], [1,2,9]]
#     '''
#     result = []
#     for label_string in strings:
#         # Remove trailing semicolon and split by this separator
#         labels = label_string[:-1].split(';')
#         # Convert all elements from string to int
#         result.append(list(map(int, labels)))
#     return result

# def onehot(y):
#     y_onehot = torch.FloatTensor(args.num_classes).zero_()
#     for i in y:
#         y_onehot[i] = 1
#     return y_onehot

# def onehot_2d(classes_list, rows):
#     result = torch.FloatTensor(rows, args.num_classes)
#     for row, classes in enumerate(classes_list):
#         classes = torch.LongTensor(classes)
#         result[row, :] = onehot(classes)
#     return result

# def validate(validation_data, model, criterion):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     topk = AverageMeter()
#     model.eval()

#     # Tensor for storing all aggregated outputs for one song (many spectrograms)
#     summed_output_classwise = torch.FloatTensor(1, args.num_classes).zero_()

#     timer_start = time.time()
#     for step, (input, target, song_path) in enumerate(validation_data):
        
#         # Adjust 2d onehot matrix height to mini-batch size
#         target = onehot_2d(string_to_class_idx(target), rows=input.shape[0])

#         if args.use_cuda:
#             target = target.cuda(async=True)
#         input_var = torch.autograd.Variable(input)
#         target_var = torch.autograd.Variable(target)
        
#         # Compute output
#         output = model(input_var)
        

#         # Sum activations class-wise
#         [summed_output_classwise.add_(row) for row in output.cpu().data]

#         # DEBUG one row for simplicity
#         print('Song path: {}\n'.format(Path(song_path[0]).stem))
#         print('Output - ', output[0])
#         print('Target: ', target[0])
        
#         # TODO fix and uncomment code below
#         # TODO figure out how to measure accuracy

#         # prec_1, prec_k = accuracy(output.data, target, topk=(1, args.top_k))
#         # top1.update(prec_1[0], input.size(0))
#         # topk.update(prec_k[0], input.size(0))

#         # Measure batch performance and update timer
#         batch_time.update(time.time() - timer_start)
#         timer_start = time.time()

#         # Print to the console and log to Tenorboard
    
#     loss = criterion(output, target_var)
#     losses.update(loss.data[0], input.size(0))
#     print_validation_step(step, len(validation_data), batch_time, losses, top1, topk)
#     log_to_tensorboard(model, step, input_var, losses, top1, topk, mode='val')
#     # Print normalized vector of answers for one song
#     print('==============================================')
#     print('Aggregated output for: {}'.format(Path(song_path[0]).stem), summed_output_classwise / torch.max(summed_output_classwise))
#     print('Target: ', target[0].view(1, -1))
#     print('==============================================')
#     print(' * Prec@1 {top1.avg:.3f} Prec@{} {topk.avg:.3f}'.format(args.top_k, top1=top1, topk=topk))
#     return top1.avg