import librosa
import soundfile as sf
import argparse
import matplotlib.image as image
from utils import printing_functions as pf
from pathlib import Path
import numpy as np
import time
import better_exceptions

parser = argparse.ArgumentParser(description='WAV to spectrograms processor')

# Paths and dirs
parser.add_argument('-i', '--input-wav-file', required=True, default='', type=str, metavar='<PATH>')
parser.add_argument('-o', '--output-dir', required=True, default='spectrograms', type=str, metavar='<PATH>')

# General
parser.add_argument('--sr', default=22050, type=int, metavar='<int>',help='momentum')
parser.add_argument('-s', '--silence-threshold', default=0.2, type=float, metavar='<float from range <0, 1> >')
parser.add_argument('-t', '--dtype', default='float32', type=str, metavar='<variable type to store numbers>')

# Spectrogram
parser.add_argument('-S', '--spec-stretch-factor', default=1.73, type=float, metavar='<coefficient>')
parser.add_argument('-H', '--spec-height', default=224, type=int, metavar='<in pixels>' )
parser.add_argument('-W', '--spec-width', default=224, type=int, metavar='<in pixels>')
parser.add_argument('-F', '--spec-max-freq', default=11025, type=int, metavar='<frequency in Hz>')

# Window
parser.add_argument('-L', '--segment-length', default=3, type=float, metavar='<seconds>')
parser.add_argument('-O', '--segment-overlap-length', default=1.5, type=float, metavar='<seconds>')

class Preprocessor:
    def __init__(self, args):
        # Paths and dirs
        self.input_path = args.input_wav_file
        self.input_file = File(self.input_path)

        self.output_dir = Path(args.output_dir)

        # General
        self.sr = args.sr
        self.silence_threshold = args.silence_threshold
        self.dtype = args.dtype

        # Spectrogram
        self.spec_strech = args.spec_stretch_factor
        self.spec_height = args.spec_height
        self.spec_width = args.spec_width
        self.max_freq = args.spec_max_freq

        # Window
        self.segment_length = args.segment_length
        self.overlap = args.segment_overlap_length


    def _audio_files_list(self, root, ext=['wav'], recurse=False):
        files = librosa.util.find_files(root, ext=ext, recurse=recurse)
        print('Found {} files in {}\n'.format(len(files), root))
        return files

    def _blocks_num(self, file):
        # TODO -> take care of segment length and overlapping frames
        return '?'

    def _to_mono(self, data):
        # 1D -> Already mono
        if len(data.shape) == 1:
            return data
        # 2D -> Stereo to mono
        elif len(data.shape) == 2:
            return np.mean(data, axis=1)
        raise TypeError(
            'Audio data must be mono or stereo. More channels are not supported!')

    def _to_spectrogram(self, y):
        ''' Transform time-series to spectrogram operations'''
        #stft_matrix = librosa.stft(y)
        #magnitute_matrix = np.abs(stft_matrix)**2
        #mel_spectrogram = librosa.feature.melspectrogram(S=magnitute_matrix, sr=self.sr, fmax=self.max_frequency)

        # TODO wrap this nicely: n_mels is adjusted for MobileNet
        mel_spectrogram = librosa.feature.melspectrogram(y, n_mels=self.spec_height)
        return librosa.power_to_db(mel_spectrogram)

    def _blocks(self):
        '''Returns a generator for iterating over the input file (buffered read)'''
        options = {
            'b_size': int(self.input_file.original_sampling_rate * self.segment_length),
            'overlap': int(self.input_file.original_sampling_rate * self.overlap),
            'dtype': self.dtype
        }
        return sf.blocks(self.input_path,
                           blocksize=options['b_size'],
                           overlap=options['overlap'],
                           dtype=options['dtype'])

    def _contains_silence(self, y):
        return (y < self.silence_threshold).all()
    
    def _dump_block_to_spectrogram(self, block, block_idx):
        escaped_name = self.input_file.path.stem
        new_name = '{}_{}.png'.format(escaped_name, block_idx)
        output_dir = self.output_dir / escaped_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = str(output_dir / new_name)
        image.imsave(output_path, self._to_spectrogram(block))

        # Alternatively store in a binary format
        # np.save(file.output_path(block_idx, output_dir), self.to_spectrogram(y))

    @pf.print_execution_time
    def convert(self):
        print('Processing {}...\n'.format(self.input_file.full_name))

        # Divide audio into blocks and read one by one (buffered read)
        blocks = self._blocks()
        for block_idx, block_data in enumerate(blocks):
            timer = time.clock()

            y = self._to_mono(block_data)
            y = librosa.resample(y, self.input_file.original_sampling_rate, self.sr * self.spec_strech)

            # Classify very silent blocks as empty -> won't generate a spectrogram
            if self._contains_silence(y):
                print("[✗] {} block contains silence only -> no spectrogram".format(block_idx))
                continue

            self._dump_block_to_spectrogram(y, block_idx)
            
            # Display updates to the console
            millis = int(round((time.clock() - timer) * 1000))
            print('[✓] {} took {}ms'.format(block_idx + 1, millis), end='\r')
        print('DONE ✓')

class File():
    '''Helper class'''
    def __init__(self, file_path):
        self.path = Path(file_path)
        self.full_name = self.path.name

        # Extract informations
        info = sf.info(file_path)
        self.original_sampling_rate = info.samplerate
        self.frames = info.frames
        self.duration = info.duration

if __name__ == '__main__':
    args = parser.parse_args()
    p = Preprocessor(args)
    p.convert()
