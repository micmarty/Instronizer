from torchvision.transforms import Compose, ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
class Dataset():
    '''
    Class which takes care of:
    - finding
    - transforming
    - splitting
    - and shuffling the data
    in given root folder
    '''
    def __init__(self, root, batch_size=100, shuffle=True, num_workers=4):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.subsets = ['train', 'test']
        self.datasets = {
            subset: ImageFolder(os.path.join(root, subset),
                                self.data_transforms()[subset])
            for subset in self.subsets
        }
        
    def data_transforms(self):
        return {
            'train': Compose([
                #transforms.RandomSizedCrop(224),
                #transforms.RandomHorizontalFlip(),
                ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': Compose([
                #transforms.Scale(256),
                #transforms.CenterCrop(224),
                ToTensor(),
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def data_loaders(self):
        return {
            subset: DataLoader(self.datasets[subset],
                          batch_size=self.batch_size,
                          shuffle=self.shuffle, 
                          num_workers=self.num_workers)
            for subset in self.subsets
        }
    
    def dataset_sizes(self):
        return {
            subset: len(self.datasets[subset])
            for subset in self.subsets
        }