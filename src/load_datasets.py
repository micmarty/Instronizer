from torchvision import transforms
from torchvision.datasets import ImageFolder

class Dataset():
    def __init__(self, root, batch_size, shuffle, num_workers):
        subsets = ['train', 'test']

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

        datasets = {
            subset: ImageFolder(os.path.join(root, subset),
                                data_transforms[subset])
            for subset in subsets
        }

        data_loaders = {
            subset: DataLoader(datasets[subset],
                          batch_size=batch_size,
                          shuffle=True, num_workers=num_workers)
            for subset in subsets
        }

    def training_data():




dataset_sizes = {
    x: len(image_datasets[x])
    for x in dataset_types
}
class_names = image_datasets['train'].classes
