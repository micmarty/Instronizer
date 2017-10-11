import models
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
#from torch import FloatTensor, max, load, save
import torch
import torch.nn as nn
from torch.autograd import Variable
import default_settings as SETTINGS
import os
from splitdataset import SplitDataset
from torchvision import transforms
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from dataset import Dataset
import utils
import argparse
from pathlib import Path
import shutil
import signal
import time

# TODO remove unneccessary imports

# Parameters ------------------------------------------
learning_rate = 0.001
batch_size = 64
num_epochs = 20
num_workers = 4
print_every = 2

model = models.LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    shutil.copyfile(filename, 'model_best.pth.tar')

@utils.print_execution_time
def train(start_epoch, train_ds, train_ds_size):
    all_losses = []

    for epoch in range(start_epoch, num_epochs):
        for i, (images, labels) in enumerate(train_ds):
            images = Variable(images)
            labels = Variable(labels)

            #print(images[0], labels[0])
            # Forward + Backward + Optimize

            # By default PyTorch doesn't flush old gradients
            # without this it would add old and new together
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % print_every == 0:
                all_losses.append(loss.data[0])
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, train_ds_size // batch_size, loss.data[0]))

        
        # TODO it weights to much, need to overwrite...
        checkpoint_name = '{}_checkpoint__epoch_{}.pth.tar'.format(
            model.__class__.__name__, epoch + 1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': model.__class__.__name__,
            'state_dict': model.state_dict(),
            'loss': loss.data[0],
            'optimizer': optimizer.state_dict(),
        }, filename=checkpoint_name)
        print('Epoch: {}.Saved checkpoint {}'.format(epoch+1,checkpoint_name))

    plt.figure()
    plt.plot(all_losses)

@utils.print_execution_time
def test(test_ds, test_ds_size):
    '''Runs the model on testing data'''
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_ds:
        images = Variable(images)
        outputs = model(images)
        
        #print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        for label, prediction in zip(labels, predicted):
            print('Label: {} | Prediction: {}'.format(label, prediction))
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the %d test images: %d %%' %
        (test_ds_size, 100 * correct / total))

def exit_gracefully_handler(signum, frame):
    '''Function called on <Ctrl + c>'''

    print('Caught SIGINT: Exiting gracefully...')

    # Save model
    save_model_decision = input('[1/2] Do you want to save the model?\n[y/N]')
    if save_model_decision is 'y':
        print('Saving model')
        #save_checkpoint()
        pass
    elif save_model_decision in ('n' or ''):
        print('Model not saved')
        pass

    # Plot loss function
    print('2. Loss function:')
    plot_loss_decision = input(
        '[2/2] Do you want to plot the loss function?\n[y/N]')
    if plot_loss_decision is 'y':
        #plot_loss()
        pass
    elif plot_loss_decision in ('n' or ''):
        pass
    exit()

def parsed_args():
    '''...'''
    parser = argparse.ArgumentParser(
        description='Run training for instrument recognition with DNN')

    parser.add_argument('-i', '--input-spectrogram-dir',
                        default='/home/miczi/Projects/single-instrument-recognizer/output/spectrograms',
                        action=utils.FullPaths,
                        type=utils.is_dir,
                        required=False,
                        help='Path to folder with dataset structure containing audio files \n \
                                (put into label-like folder name, e.g. barking.wav inside dogs/, miau.wav inside cats/)')

    parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    return parser.parse_args()

def load_datasets(args):
    '''...'''
    dataset = Dataset(root=args.input_spectrogram_dir,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)

    ds = dataset.data_loaders()
    ds_sizes = dataset.dataset_sizes()

    train_ds, test_ds = ds['train'], ds['test']
    train_ds_size, test_ds_size = ds_sizes['train'], ds_sizes['test']

    print('Dataset sizes:')
    print(' - training: {} images'.format(train_ds_size))
    print(' - testing:  {} images'.format(test_ds_size))

    return ((train_ds, train_ds_size), (test_ds, test_ds_size))

def resume_from_checkpoint(checkpoint_path):
    if not Path(checkpoint_path).is_file():
        print("=> No checkpoint found at '{}'".format(checkpoint_path))
        return 0
    
    
    # Unpack checkpoint file into variables
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # Restore model and optimizer state
    print("=> Loading checkpoint '{}'".format(checkpoint_path))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print("=> Resuming from checkpoint: '{}' (epoch {})"
          .format(checkpoint_path, start_epoch))
    return start_epoch

if __name__ == '__main__':
    # Set up handler which fires up on ctrl + c
    signal.signal(signal.SIGINT, exit_gracefully_handler)
    args = parsed_args()
    (train_ds, train_ds_size), (test_ds, test_ds_size) = load_datasets(args)

    
    start_epoch = resume_from_checkpoint(args.resume)

    train(start_epoch, train_ds, train_ds_size)
    test(test_ds, test_ds_size)

