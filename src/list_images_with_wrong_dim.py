"""
Lists all the files with dimensions other than specified,
so you can remove them before the training starts.

Useful with image datasets, it saves you from crashing
in the middle of training process.
"""
from pathlib import Path
from PIL import Image
import argparse
import utils

parser = argparse.ArgumentParser(description='Lists all the files with dimensions other than specified, \
                                 so you can remove them before the training starts. \
                                 Useful with image datasets, it saves you from crashing \
                                 in the middle of training process.')

parser.add_argument('dir',
                    action=utils.FullPaths,
                    type=utils.is_dir,
                    help='Directory with images to check')
args = parser.parse_args()

SEARCHING_ROOT_DIR = args.dir
GOOD_WIDTH, GOOD_HEIGHT = 224, 224
IMAGE_EXTENSION = '.png'

print('SEARCHING_ROOT_DIR =', SEARCHING_ROOT_DIR)
print('GOOD_WIDTH, GOOD_HEIGHT = {}, {}'.format(GOOD_WIDTH, GOOD_HEIGHT))
print('IMAGE_EXTENSION = \'{}\'\n'.format(IMAGE_EXTENSION))

class ProgressBar():
    def __init__(self):
        self.full_char, self.empty_char = '█', '░'
        self.counter = 1
        self.direction = 1
        self.max_counter = 40
        self.update_freq = 40

    def updated(self):
        if self.counter in [0, self.max_counter]:
            self.direction = -self.direction

        if self.direction > 0:
            self.counter += 1
        else:
            self.counter -= 1
        progress_bar = self.empty_char * (self.counter - 1) + \
            self.full_char * 10 + \
            self.empty_char * (self.max_counter - self.counter)
        return progress_bar


# Execution start
files = Path(SEARCHING_ROOT_DIR).glob('**/*' + IMAGE_EXTENSION)
invalid_images = []
progress_bar = ProgressBar()
idx = 0

for file_path in files:
    # Load image, get its dimensions
    im = Image.open(file_path)
    width, height = im.size

    if idx % progress_bar.update_freq == 0:
        print('\rSearching for files with invalid dimensions... ',
              progress_bar.updated(), end='')

    if width is not GOOD_WIDTH or GOOD_HEIGHT is not 224:
        invalid_images.append(file_path)
    idx += 1

if invalid_images:
    print('\nInvalid images: ')
    print('Just copy the text below and run with `rm`\n')
    [print('\"{}\"'.format(path)) for path in invalid_images]
    exit(0)
else:
    print('\nNo invalid images was found')
    exit(-1)
