import torch
from pathlib import Path
import dataset_finder_copy as df
import argparse
from mobilenet_copy import MobileNet


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input', default=None, metavar='DIR', help='path to dir containing .npy spectrograms')
#     return parser.parse_args()

def load_data_from_folder(path):
    dataset = df.SpecFolder(path)
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=3,
                                       shuffle=False,
                                       num_workers=3)

def run(input):
    #args = parse_args()
    model = MobileNet(num_classes=11)
    
    validation_data = load_data_from_folder(input)
    aggregated_output = None
    for step, (input, target) in enumerate(validation_data):
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Compute output
        output = model(input_var)
        print(output)
        aggregated_output = torch.sum(output, dim=0)  # size = [1, ncol]
        print(aggregated_output)
        maxe = aggregated_output.max(0)
        print("Max value", maxe[0], maxe[1])
    return maxe[1].data[0]
