import model
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch import FloatTensor, max
import torch.nn as nn
from torch.autograd import Variable
from torch import save
import default_settings as SETTINGS
import os
from splitdataset import SplitDataset
from torchvision import transforms
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from dataset import Dataset
from utils import print_execution_time
# TODO remove unneccessary imports

# Parameters
learning_rate = 0.01
batch_size = 100
num_epochs = 10
num_workers = 4
print_every = 10

# Dataset phase
dataset = Dataset(root=SETTINGS.PATHS['OUTPUT_SPECTROGRAM_DIR'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
ds = dataset.data_loaders()
ds_sizes = dataset.dataset_sizes()

train_ds, test_ds = ds['train'], ds['test']
train_ds_size, test_ds_size = ds_sizes['train'], ds_sizes['test']

print('Dataset sizes:')
print(' - training: {} images'.format(train_ds_size))
print(' - testing:  {} images'.format(test_ds_size))

cnn = model.AlexNet() #model.LeNet()
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

@print_execution_time
def train():
    all_losses = []

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_ds):
            images = Variable(images)
            labels = Variable(labels)

            #print(images[0], labels[0])
            # Forward + Backward + Optimize

            # By default PyTorch doesn't flush old gradients
            # without this it would add old and new together
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % print_every == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, train_ds_size // batch_size, loss.data[0]))

            if (i + 1) % 1000 == 0:
                all_losses.append(loss.data[0])
    plt.figure()
    plt.plot(all_losses)
    

train()



# # Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_ds:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the %d test images: %d %%' %
      (test_ds_size, 100 * correct / total))

# Save the Trained Model
save(cnn.state_dict(), 'cnn.pkl')
plt.show()
