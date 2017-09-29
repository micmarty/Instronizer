import torch.utils.data as torch_data
import librosa
import os

class InstrumentsDataset(torch_data.Dataset):
    datasets_folder = 'datasets'
    raw_folder = 'raw'
    processed_folder = 'processed'
    
    def __init__(self, root, dataset_name):
        self.working_dataset_path = os.path.join(root, self.datasets_folder, dataset_name)
        self.input = []
        self.labels = []

        # TODO check if dataset exists
        # raise RuntimeError('Dataset not found! \n' +
        #                     'Make sure to download it and put into "{}" folder.\n'.format(self.datasets_folder) + 
        #                     'TODO allow for automatic download')
           


    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
