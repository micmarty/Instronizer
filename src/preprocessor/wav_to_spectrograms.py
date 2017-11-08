''' Created by Michał Martyniak 

Description:
This script supports input like parameters as following: 
- full path to a single wav file
- dir path to IRMAS dataset
- path to dir containing wav files

The destination folder doesn't have to exist (it will be automatically created)

Example with IRMAS dataset (preprocess training data):
python src/wav_to_spectrograms.py -i <...>/train --irmas -o <...>/spectrograms

Example with a single wav file (will create separate folder in <out_dirpath>)
python src/wav_to_spectrograms.py -i <...>/hulu.wav -o .

Example with dir containing wav files:
python src/wav_to_spectrograms.py -i <...>/train/cel -o <...>/cello_spectrograms
'''

import librosa
import soundfile as sf
import argparse
from pathlib import Path, PurePath
import numpy as np
import time
import scipy
import better_exceptions

# Custom imports
from classifier.utils import printing_functions as pf

parser = argparse.ArgumentParser(description='WAV to spectrograms processor')

# Paths and dirs
parser.add_argument('-i', '--input', required=True, default='', type=str, metavar='<PATH>')
parser.add_argument('-o', '--output-dir', required=True, default='spectrograms', type=str, metavar='<PATH>')
parser.add_argument('--irmas', action='store_true', help='output dir structure will preserved with data type and labels (e.g. train/cel/<spectrogram>.npy')

# General
parser.add_argument('--sr', default=22050, type=int, metavar='<int>',help='momentum')
parser.add_argument('-s', '--silence-threshold', default=-0.9, type=float, metavar='<float from range <-1, 1> >')
parser.add_argument('-t', '--dtype', default='float32', type=str, metavar='<variable type to store numbers>')

# Spectrogram
parser.add_argument('-S', '--spec-stretch-factor', default=1, type=float, metavar='<coefficient>')
parser.add_argument('-H', '--spec-height', default=224, type=int, metavar='<in pixels>' )
parser.add_argument('-W', '--spec-width', default=224, type=int, metavar='<in pixels>')
parser.add_argument('-F', '--spec-max-freq', default=11025, type=int, metavar='<frequency in Hz>')

# Window
parser.add_argument('-L', '--segment-length', default=3, type=float, metavar='<seconds>')
parser.add_argument('-O', '--segment-overlap-length', default=1.5, type=float, metavar='<seconds>')
parser.add_argument('--start', default=None, type=float, metavar='<seconds>')
parser.add_argument('--end', default=None, type=float, metavar='<seconds>')

class Preprocessor:
    def __init__(self, args):
        # Paths and dirs
        self.use_irmas_folder_structure = args.irmas

        # Check if we need to process a single or multiple files
        if Path(args.input).is_dir():
            # Input was a directory path, so multiple files
            if args.irmas:
                self.input = self._audio_files_list(args.input, ext = ['wav'], recurse=True)
            else:
                self.input = self._audio_files_list(args.input, ext=['wav'], recurse=False)
        else:
            # Single file
            self.input = args.input

        self.output_dir = Path(args.output_dir)

        # General
        self.sr = args.sr
        self.silence_threshold = args.silence_threshold
        self.dtype = args.dtype

        # Spectrogram
        self.spec_stretch = args.spec_stretch_factor # Unused
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
                sr=self.sr,  
                fmax=self.max_freq, 
                n_mels=self.spec_height)
        S = librosa.logamplitude(S)
        S = scipy.misc.imresize(S, (self.spec_height, self.spec_width), interp='bilinear')
        return librosa.util.normalize(S) 

    def _contains_silence(self, data):
        return np.mean(data) < self.silence_threshold
    
    def _output_dir(self, songpath):
        '''
        Chooses a final path for each spectrogram and creates missing folders
        Returns an existing path (Path object)
        '''

        if self.use_irmas_folder_structure:
            # This flag assumes IRAMAS dataset folder structure (e.g. test/cel)
            input_path = Path(songpath)
            # e.g. 'cel'
            label_name = input_path.parent.name
            # e.g. 'train'
            dataset_name = input_path.parent.parent.name

            output_dir = self.output_dir / dataset_name / label_name
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        elif self.create_dir_for_specs:
            # Creates separate folder for one song
            output_dir = self.output_dir / PurePath(songpath).stem
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        else:
            # When you want to just put everything in specified folder
            self.output_dir.mkdir(parents=False, exist_ok=True)
            return output_dir
        
    def _dump_spectrogram(self, y, songpath, counter):
        spectrogram = self._to_spectrogram(y)
        if self._contains_silence(spectrogram):
            print('[✗] Segment {} contains mostly silence, skipping...'.format(counter))
            return False
        songname = PurePath(songpath).stem
        new_name = '{}_{}.npy'.format(songname, counter)
        output_path = str(self._output_dir(songpath) / new_name)

        np.save(output_path, spectrogram)
        return True
        # Alternatively:
        #image.imsave(output_path, self._to_spectrogram(block))


    @pf.print_execution_time
    def process(self, time_range):
        if isinstance(self.input, str):
            # Single file
            self.create_dir_for_specs = True
            self._convert(self.input, time_range)
        elif isinstance(self.input, list):
            # Many files
            self.create_dir_for_specs = True
            for filepath in self.input:
                self._convert(filepath, time_range=(None, None))
        else:
            print('Skipping')

    #@pf.print_execution_time
    def _resample(self, y, from_sr):
        return librosa.resample(y, from_sr, self.sr, res_type='kaiser_fast')

    def _convert(self, wav_file_path, time_range):
        info = File(wav_file_path)
        y, original_sr = sf.read(wav_file_path)
        y = self._to_mono(y)

        # IMPORTANT!
        # Resampling can be very expensive (takes about 200ms additional time for computations on 3s audio)
        y =  self._resample(y, from_sr=original_sr)

        generated_specs_counter = 0
        # TODO we can make double-check if time range is valid (e.g. not exceeding duration or starts before 0)
        if isinstance(time_range, tuple):
            offset, end_boundary = time_range
            if offset is None:
                offset = 0
            if end_boundary is None:
                end_boundary = info.duration
        else:
            raise Exception('time_range type error', 'must be tuple of 2 values')
            
        while offset + self.segment_length <= round(end_boundary):
            timer = time.clock()

            start = int(offset * self.sr)
            end = int(start + self.segment_length * self.sr)

            # Save spectrogram
            success = self._dump_spectrogram(y[start:end], songpath=wav_file_path, counter=generated_specs_counter)

            # Prepare for the next iteration
            millis = int(round((time.clock() - timer) * 1000))
            if success:
                print('[✓] Segment {} (range => {} - {} / {}) took {}ms'.format(generated_specs_counter, 
                                                                                start, end, len(y), millis))

            generated_specs_counter += 1
            offset += self.overlap
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
    p.process(time_range=(args.start, args.end))
    exit(0)
