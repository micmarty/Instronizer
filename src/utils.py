import argparse
import os
import default_settings as SETTINGS

class FullPaths(argparse.Action):
    '''Expand user and relative-paths'''
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))


def is_dir(dirname):
    '''Checks if a path is an actual directory'''
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname

def print_parameters(args):
    '''TODO'''
    print('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        print('\t{}={}'.format(attr.upper(), value))


def parse_args():
    '''Take input from command line'''
    parser = argparse.ArgumentParser(
        description='Instrument recognition with DNN in PyTorch')

    parser.add_argument('-i', '--input-dataset-dir',
                        action=FullPaths,
                        type=is_dir,
                        required=False,
                        default=SETTINGS.PATHS['INPUT_DATASET_DIR'],
                        help=SETTINGS.HELP['INPUT_DATASET_DIR'])

    parser.add_argument('-o', '--output-spectrograms-dir',
                        action=FullPaths,
                        type=is_dir,
                        required=False,
                        default=SETTINGS.PATHS['OUTPUT_SPECTROGRAM_DIR'],
                        help=SETTINGS.HELP['OUTPUT_SPECTROGRAM_DIR'])
    return parser.parse_args()
