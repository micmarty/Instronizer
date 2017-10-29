# Original source code: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
'''
It's responsible for finding spectrograms in given path and extracting their labels
'''
import torch
import numpy as np
from pathlib import Path
import os
import os.path

def is_spec_file(filename):
    return filename.endswith('.npy')

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx



def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_spec_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images

def default_loader(path):
    '''
    It loads numpy 2d spectrogram matrix and creates fake 3d tensor
    
    [PyTorch BUG]: See the return value:
    When .double() it says "expected DoubleTensor, got FloatTensor".
    When .float() it works just fine, WTF!
    '''

    spectrogram = np.load(path)
    spec_3d = np.stack((spectrogram,) * 3)
    return torch.from_numpy(spec_3d).float()


def val_targets(path):
    '''
    Takes path to txt file containing labels (one per line)
    This txt file has a name identical with folder containing spetrograms
    Returns: list of strings (labels)
    '''
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        # Remove whitespaces
        return [line.strip() for line in lines]


def irmas_classes():
    classes = ['cel', 'cla', 'flu', 'gac', 'gel',
               'org', 'pia', 'sax', 'tru', 'vio', 'voi']
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_val_dataset(dir, class_to_idx):
    dir = Path(dir)
    spec_target_pairs = []

    # For every file or directory inside dir
    for content in dir.iterdir():
        
        # If it's a txt file, extract all labels
        if content.is_file() and content.suffix == '.txt':
            classes = val_targets(str(content))
            
            # Create a path that should point to a folder with same name as txt file
            spectrogram_dir = content.parent / content.stem

            for spectrogram in spectrogram_dir.iterdir():
                # If it's a .npy file, then pair its path with matching labels
                if spectrogram.is_file() and spectrogram.suffix == '.npy':
                    # Construct string containing class indexes
                    # It's a trick that fools PyTorch dataloader, so we won't loose data
                    # Classes are separated by a colon
                    classes_string = ''
                    for class_name in classes:
                        classes_string += '{};'.format(class_to_idx[class_name])
                    pair = (str(spectrogram), classes_string)
                    spec_target_pairs.append(pair)
    return spec_target_pairs

class SpecFolder(torch.utils.data.Dataset):
    """A generic data loader where the data is like:

        root/cel/xxx.npy
        root/cel/xxy.npy
        root/cel/xxz.npy

        root/voi/123.npy
        root/voi/nsdf3.npy
        root/voi/asd932_.npy

        Args:
        TODO adapt this description for spectrogram loader

        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        specs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        specs = make_dataset(root, class_to_idx)
        if len(specs) == 0:
            raise(RuntimeError("Found 0 spectrograms in subfolders of: " + root + "\n" + \
                               "Supported spectrogram extensions are: .npy"))

        self.root = root
        self.specs = specs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.specs[index]
        spec = self.loader(path)
        if self.transform is not None:
            spec = self.transform(spec)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return spec, target

    def __len__(self):
        return len(self.specs)


class ValSpecFolder(torch.utils.data.Dataset):
    '''
    It expects that:
    - root contains folders with unique names
    - root contains .txt files (each containg any number of labels, one per line)
    - these .txt files should have names matching existing folders, 
    e.g. aurora.txt, containing: 
    ```
    voi
    pia
    gac
    ```
    +
    <root>/aurora/ containing .npy spectrograms

    '''
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        classes, class_to_idx = irmas_classes()
        specs = make_val_dataset(root, class_to_idx)
        if len(specs) == 0:
            raise(RuntimeError("Found 0 spectrograms in subfolders of: " + root + "\n" +
                               "Supported spectrogram extensions are: .npy"))

        self.root = root
        self.specs = specs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a list of strings (labels)
        """
        path, target = self.specs[index]
        spec = self.loader(path)
        if self.transform is not None:
            spec = self.transform(spec)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return spec, target

    def __len__(self):
        return len(self.specs)
