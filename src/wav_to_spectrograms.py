''' Created by Michał Martyniak 

Example:

<filepath> is an existing path to a single wav file
<dirpath> is an existing directory path with many wav files
This script adapts to provided input paths, just give it a try

<out_dirpath> is the destination and doesn't have to exist (it will be automatically created)

python src/wav_to_spectrograms.py -i <filepath_or_dirpath> -o <out_dirpath>
'''

import librosa
import soundfile as sf
import argparse
import matplotlib.image as image
from utils import printing_functions as pf
from pathlib import Path, PurePath
import numpy as np
import time
import better_exceptions

parser = argparse.ArgumentParser(description='WAV to spectrograms processor')

# Paths and dirs
parser.add_argument('-i', '--input', required=True, default='', type=str, metavar='<PATH>')
parser.add_argument('-o', '--output-dir', required=True, default='spectrograms', type=str, metavar='<PATH>')

# General
parser.add_argument('--sr', default=22050, type=int, metavar='<int>',help='momentum')
parser.add_argument('-s', '--silence-threshold', default=-0.9, type=float, metavar='<float from range <-1, 1> >')
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

        # Check if we need to process a single or multiple files
        if Path(args.input).is_dir():
            # Input was a directory path, so multiple files
            self.input = self._audio_files_list(args.input, ext = ['wav'])
        else:
            # Single file
            self.input = args.input

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
        S = librosa.feature.melspectrogram(y,  
                sr=self.sr * self.spec_strech,  
                fmax=self.max_freq * self.spec_strech, 
                n_mels=self.spec_height)
        S = librosa.logamplitude(S) 
        return librosa.util.normalize(S) 

    # def _blocks(self):
    #     '''Returns a generator for iterating over the input file (buffered read)'''
    #     options = {
    #         'b_size': int(self.input_file.original_sampling_rate * self.segment_length),
    #         'overlap': int(self.input_file.original_sampling_rate * self.overlap),
    #         'dtype': self.dtype
    #     }
    #     return sf.blocks(self.input_path,
    #                        blocksize=options['b_size'],
    #                        overlap=options['overlap'],
    #                        dtype=options['dtype'])

    def _contains_silence(self, data):
        return np.mean(data) < self.silence_threshold
    
    def _dump_spectrogram(self, y, input_file_basename, counter):
        spectrogram = self._to_spectrogram(y)
        if self._contains_silence(spectrogram):
            print('[✗] Segment {} contains mostly silence, skipping...'.format(counter))
            return

        new_name = '{}_{}.npy'.format(input_file_basename, counter)

        if self.create_dir_for_specs:
            output_dir = self.output_dir / input_file_basename
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = self.output_dir
            output_dir.mkdir(parents=False, exist_ok=True)

        output_path = str(output_dir / new_name)
        np.save(output_path, spectrogram)

        # Alternatively:
        #image.imsave(output_path, self._to_spectrogram(block))


    @pf.print_execution_time
    def process(self):
        if isinstance(self.input, str):
            # Single file
            self.create_dir_for_specs = True
            self._convert(self.input)
        elif isinstance(self.input, list):
            self.create_dir_for_specs = False
            for filepath in self.input:
                self._convert(filepath)
        else:
            print('Skipping')

    @pf.print_execution_time
    def _resample(self, y, from_sr):
        return librosa.resample(y, from_sr, self.sr * self.spec_strech, res_type='kaiser_fast')

    def _convert(self, wav_file_path):
        info = File(wav_file_path)
        y, original_sr = sf.read(wav_file_path)
        y = self._to_mono(y)

        # IMPORTANT!
        # Resampling is very expensive (takes about 200ms additional time for computations on 3s audio).
        # Disabling it will impact IRMAS dataset, because spectrograms won't be stretched to 224x224 (as mobilenet requires).
        # The only disadvantage is that we take "only" 114,438 from 132,299 (87%) existing frames in a 3s audio clip
        #
        # y =  self._resample(y, from_sr=original_sr)

        generated_specs_counter = 0
        offset = 0
        while offset + self.segment_length <= round(info.duration):
            timer = time.clock()

            # Calculations
            # Watch out! Original audio samplerate is different than self.sr
            # Example 48kHz -> 1s, 22kHz * stretch -> 0.7s
            # The reason is to get 224x224 spectrogram without stretching!
            frames = int(self.sr * self.spec_strech)
            start = int(offset * frames)
            end = int(start + self.segment_length * frames)

            # Save spectrogram
            

            ###
            self._dump_spectrogram(y[start:end], PurePath(wav_file_path).stem, generated_specs_counter)

            # Prepare for the next iteration
            millis = int(round((time.clock() - timer) * 1000))
            print('[✓] Segment {} (range => {} - {} / {}) took {}ms'.format(generated_specs_counter, offset*frames, offset * frames + self.segment_length * frames, info.frames, millis), end='\r')

            generated_specs_counter += 1
            offset += self.segment_length
        print('\nDONE ✓')

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
    p.process()
