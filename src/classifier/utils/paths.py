'''Currently not in use (in favour of mobilenet_training.py)'''
import argparse
import os
import time
import datetime

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


def is_file(file_path):
    '''Checks if a path is an actual file'''

    if not os.path.isfile(file_path):
        msg = "{0} is not a file".format(file_path)
        raise argparse.ArgumentTypeError(msg)
    else:
        return file_path
