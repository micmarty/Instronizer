import model
import torch.optim as optim
from torch.utils.data import DataLoader


from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch import FloatTensor
import torch.nn as nn
from torch.autograd import Variable

import default_settings as SETTINGS

learning_rate = 0.01
momentum = 0.5

num_epochs = 5
batch_size = 100
num_workers = 4


train_dataset = ImageFolder(root=SETTINGS.PATHS['OUTPUT_SPECTROGRAM_DIR'], transform=ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)


cnn = model.LeNet()
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)


def train():
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)

            print(images[0], labels[0])
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                    % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))

train()

# Test the Model
# cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
# correct = 0
# total = 0
# for images, labels in test_loader:
#     images = Variable(images)
#     outputs = cnn(images)
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels).sum()

# print('Test Accuracy of the model on the 10000 test images: %d %%' %
#       (100 * correct / total))

# # Save the Trained Model
# torch.save(cnn.state_dict(), 'cnn.pkl')
