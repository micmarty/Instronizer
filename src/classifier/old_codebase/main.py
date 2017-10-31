import utils

# Currently not in use (in favour of mobilenet_training.py)

if __name__ == '__main__':
    ARGS = utils.parse_args()
    utils.print_parameters(ARGS)



    # TODO maybe use onehot in the future?

    # dataset_classes_num = len(data.classes)
    # y_as_onehot = FloatTensor(1, dataset_classes_num)

    # y_as_onehot.zero_()
    # y_as_onehot.scatter_(dim=0, index=y, src=1)
    

#     nb_classes = 3
#     targets = np.array([[y]]).reshape(-1)
#     y_as_onehot = np.eye(nb_classes)[targets]
# >> > one_hot_targets


    

# import torch.utils.data as torch_data
# import librosa
# import os

# class InstrumentsDataset(torch_data.Dataset):
#     datasets_folder = 'datasets'
#     raw_folder = 'raw'
#     processed_folder = 'processed'
    
#     def __init__(self, root, dataset_name):
#         self.working_dataset_path = os.path.join(root, self.datasets_folder, dataset_name)
#         self.input = []
#         self.labels = []

#         # TODO check if dataset exists
#         # raise RuntimeError('Dataset not found! \n' +
#         #                     'Make sure to download it and put into "{}" folder.\n'.format(self.datasets_folder) + 
#         #                     'TODO allow for automatic download')
           


#     def __getitem__(self, index):
#         pass

#     def __len__(self):
#         pass
