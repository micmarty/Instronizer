import torch
from pathlib import Path
import argparse

# Absolute imports
from classifier.models.mobilenet import MobileNet
from classifier import dataset_loader as dl

def load_data_from_folder(path):
    dataset = dl.SpecFolder(path, direct=True)
    # There are 3 spectrograms from 6s window
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=3,
                                       shuffle=False,
                                       num_workers=3)

def run(input):
    model = MobileNet(num_classes=11)
    # TODO load checkpoint
    
    validation_data = load_data_from_folder(input)
    aggregated_output = None

    # This loop executes once
    for step, (input, target) in enumerate(validation_data):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Compute output
        output = model(input_var)

        # Sum outputs for each instrument
        aggregated_output = torch.sum(output, dim=0)  # size = [1, ncol]

        max_value, max_value_idx = aggregated_output.max(0)

        print('Output: ', output.data)
        print('Instrument class-wise activation sum', aggregated_output)
        print('Max: {}, instrument_idx: {}'.format(max_value, max_value_idx))
    # Cast tensor to python int type
    return max_value_idx.data[0] 
