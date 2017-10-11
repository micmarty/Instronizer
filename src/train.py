import model
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
# TODO remove unneccessary imports

# Parameters ------------------------------------------
learning_rate = 0.01
batch_size = 100
num_epochs = 10
num_workers = 4
print_every = 10
#-------------------------------------------------------

# Parse input parameters -------------------------------
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
args = parser.parse_args()



#-------------------------------------------------------

# Dataset phase ----------------------------------------
dataset = Dataset(root=args.input_spectrogram_dir, batch_size=batch_size, shuffle=True, num_workers=num_workers)
ds = dataset.data_loaders()
ds_sizes = dataset.dataset_sizes()

train_ds, test_ds = ds['train'], ds['test']
train_ds_size, test_ds_size = ds_sizes['train'], ds_sizes['test']

print('Dataset sizes:')
print(' - training: {} images'.format(train_ds_size))
print(' - testing:  {} images'.format(test_ds_size))
#-------------------------------------------------------

# Training ---------------------------------------------
model = model.AlexNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# optionally resume from a checkpoint
start_epoch = 0
if args.resume:
    if Path(args.resume).is_file():
        print("=> Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> Loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(args.resume))
else:
    print('=> Starting training with fresh model')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    shutil.copyfile(filename, 'model_best.pth.tar')

@utils.print_execution_time
def train():
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
    

train()
#---------------------------------------------------------

# Test the Model -----------------------------------------
model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_ds:
    images = Variable(images)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the %d test images: %d %%' %
      (test_ds_size, 100 * correct / total))

# Save the Trained Model
torch.save(model.state_dict(), 'model.pkl')
plt.show()
