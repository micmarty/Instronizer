import argparse
import os
import default_settings as SETTINGS
import time
import datetime
import matplotlib.pyplot as plt

def print_execution_time(function):
    '''Decorator which measures function's execution time

    Just add @print_execution_time above your function definition
    '''
    def wrapper(*args, **kw):
        start_time = time.clock()
        function(*args, **kw)
        formatted_time_took = datetime.timedelta(seconds=(time.clock() - start_time))
        print('Funtion {} took: {}'.format(
            function.__name__, formatted_time_took))

    return wrapper


class FullPaths(argparse.Action):
    '''Expand user and relative-paths'''

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))



def print_parameters(args):
    '''Prints arguments from command line to the console'''

    print('Parameters:')
    for attr, value in sorted(args.__dict__.items()):
        print('\t{}={}'.format(attr.upper(), value))

def is_dir(dirname):
    '''Checks if a path is an actual directory'''

    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname

def parse_args():
    '''Takes the input from command line and returns argparser object'''

    parser = argparse.ArgumentParser(
        description=SETTINGS.STRINGS['ARG_PARSER_DESCRIPTION'])

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





# TODO it used to work, but now it doesn't

label_names = [
    'cello',
    'piano'
]

def plot_images(images, cls_true, cls_pred=None):

    assert len(images) == len(cls_true) == 9

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot the image
        ax.imshow(images[i, :, :, :], interpolation='spline16')
        # get its equivalent class name
        cls_true_name = label_names[cls_true[i]]

        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name)

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
