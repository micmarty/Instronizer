"""
Lists all the files with dimensions other than specified,
so you can remove them before the training starts.

Useful with image datasets, it saves you from crashing
in the middle of training process.
"""
from pathlib import Path
import argparse
import utils
from PIL import Image

# Parse input
parser = argparse.ArgumentParser(description='Lists all the files with dimensions other than specified, \
                                 so you can remove them before the training starts. \
                                 Useful with image datasets, it saves you from crashing \
                                 in the middle of training process.')
# required=True by default
parser.add_argument('dir',
                    action=utils.FullPaths,
                    type=utils.is_dir,
                    help='Directory with images to check')
args = parser.parse_args()

# CONSTANTS
SEARCHING_ROOT_DIR = args.dir
GOOD_WIDTH, GOOD_HEIGHT = 224, 224
IMAGE_EXTENSION = '.png'

# Log parameters
print('SEARCHING_ROOT_DIR =', SEARCHING_ROOT_DIR)
print('GOOD_WIDTH, GOOD_HEIGHT = {}, {}'.format(GOOD_WIDTH, GOOD_HEIGHT))
print('IMAGE_EXTENSION = \'{}\'\n'.format(IMAGE_EXTENSION))

class ProgressBar():
    '''Progressbar logic'''
    def __init__(self):
        self.full_char, self.empty_char = '█', '░'
        self.counter = 1
        self.bar_width = 10     
        self.direction = 1      # -1 means LEFT, 1 means RIGHT
        self.max_counter = 40   # bar length
        self.update_freq = 40   # increase to slow down

    def updated(self):
        '''Animate bar, updating the state'''

        # If on left or right border
        if self.counter in [0, self.max_counter]:
            self.direction = -self.direction

        # Move towards defined direction
        if self.direction > 0:
            self.counter += 1
        else:
            self.counter -= 1

        # Concatenate <empty>(n-1) + <full>(bar_width) + <empty>(max_counter - n)
        # Resulting in e.g. '      ||||||||                    '
        progress_bar_string = self.empty_char * (self.counter - 1) + \
            self.full_char * self.bar_width + \
            self.empty_char * (self.max_counter - self.counter)
        return progress_bar_string


# Execution start
files_paths = Path(SEARCHING_ROOT_DIR).glob('**/*' + IMAGE_EXTENSION)
progress_bar = ProgressBar()
invalid_images = []
loop_idx = 0

for file_path in files_paths:
    # Load an image, get its dimensions
    im = Image.open(file_path)
    width, height = im.size

    # Display updated progress bar
    if loop_idx % progress_bar.update_freq == 0:
        print('\rSearching for files with invalid dimensions... ',
              progress_bar.updated(), end='')

    # Collect invalid image path
    if (width is not GOOD_WIDTH) or (height is not GOOD_HEIGHT):
        invalid_images.append(file_path)

    loop_idx += 1

# Show results
if invalid_images:
    print('\nInvalid images: ')
    print('Just copy the text below and run with `rm`.\n')
    [print('\"{}\"'.format(path)) for path in invalid_images]
    exit(0)
else:
    print('\nNo invalid images was found.')
    exit(-1)
