import model
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch import FloatTensor
import torch.nn as nn
from torch.autograd import Variable
from torch import save
import default_settings as SETTINGS
import os
from splitdataset import SplitDataset
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
learning_rate = 0.01
batch_size = 10

num_epochs = 20
num_workers = 4

data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomSizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        #transforms.Scale(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
dataset_types = ['train', 'test']

image_datasets = {
        x: ImageFolder(os.path.join(SETTINGS.PATHS['OUTPUT_SPECTROGRAM_DIR'], x), data_transforms[x])
        for x in dataset_types
    }

dataloders = {
            x: DataLoader(image_datasets[x], 
                batch_size=batch_size,
                shuffle=True, num_workers=num_workers)
            for x in dataset_types
        }

dataset_sizes = {
        x: len(image_datasets[x]) 
        for x in dataset_types
}
class_names = image_datasets['train'].classes


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp, interpolation='nearest')
    if title is not None:
        plt.title(title)
    plt.show()


# Get a batch of training data
inputs, classes = next(iter(dataloders['train']))

# Make a grid from batch
out = make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

exit()











def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, axes=(1, 2, 0)), interpolation='nearest')
    plt.show()
    

label_names = [
    'cello',
    'piano'
]


def plot_images(dataset, subset_range, predictions=None):
    assert isinstance(subset_range, tuple)

    ncols = 4
    nrows = (subset_range[1] - subset_range[0]) // ncols
    
    # Create figure with sub-plots.
    fig, axes = plt.subplots(nrows, ncols, figsize=(7, 8))

    for i, ax in enumerate(axes.flat):
        # plot the image
        image = dataset[i][0].numpy()
        ax.imshow(np.transpose(image, axes=(1, 2, 0)),
                  interpolation='nearest')
        
        # get its equivalent class name
        label_id = dataset[i][1]
        cls_true_name = label_names[label_id]

        if predictions is None:
            xlabel = "{0} ({1})".format(cls_true_name, label_id)
        else:
            cls_pred_name = label_names[predictions[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name)

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

dataset = ImageFolder(root=SETTINGS.PATHS['OUTPUT_SPECTROGRAM_DIR'], transform=ToTensor())
dataset = SplitDataset(dataset, partitions={'train': 0.8, 'test': 0.2}, initial_partition='train')


train_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=num_workers)

plot_images(dataset, subset_range=(0, 20))
exit()

cnn = model.LeNet()
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)


def train():
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate():
            images = Variable(images)
            labels = Variable(labels)

            #print(images[0], labels[0])
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            #if (i + 1) % 1 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

train()

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images)
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' %
      (100 * correct / total))

# Save the Trained Model
save(cnn.state_dict(), 'cnn.pkl')
