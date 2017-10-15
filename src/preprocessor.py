import librosa
import soundfile as sf
import numpy as np

import matplotlib.image as image
import matplotlib.pyplot as plt

import time
from pathlib import PurePath, Path
import utils
import argparse

import default_settings as SETTINGS

class PreprocessingSettings():

    def __init__(self):    
        # Target sampling rate
        self.sr = 22050

        # Given in seconds
        self.segment_duration = 5.2
        self.segment_overlap = self.segment_duration // 2

        # not in use
        #self.segment_frames_num = int(self.sr * self.segment_duration)
        #self.overlapping_frames_num = int(self.sr * self.segment_overlap)

        self.dtype = 'float64'
        self.silence_threshold = 0.01
        # From Nyquist theorem -> we're able to register frequencies up to 11kHz (when sampling rate is set up at 22kHz)
        self.max_frequency = self.sr // 2   

class File():
    def __init__(self, file_path):
        self.input_path = PurePath(file_path)
        self.full_name = self.input_path.name

        # Extract informations
        info = sf.info(file_path)
        self.original_sampling_rate = info.samplerate
        self.frames = info.frames
        self.duration = info.duration
    
    def output_path(self, block_index, output_dir, ext='.png', without_folder_structure=False):
        # e.g. 'cello'
        label_name = self.input_path.parent.name
        # e.g. 'train'
        dataset_name = self.input_path.parent.parent.name

        # e.g. 'cello_suite_256'
        new_name = '{}_{}'.format(self.input_path.stem, block_index)

        # Add the target extension
        output_name = self.input_path.with_name(new_name).with_suffix(ext).name
        output_dir = Path(output_dir)

        # Join paths together
        # TODO Change comment!
        # e.g. '.../train/cello/'
        if without_folder_structure:
            final_dir = output_dir
        else:
            final_dir = output_dir / dataset_name / label_name



        # Create missing folders all the way up
        final_dir.mkdir(parents=True, exist_ok=True)

        # e.g. '.../train/cello/cello_suite_256.png'
        return str(final_dir / output_name)

class Preprocessor(PreprocessingSettings):
    def __init__(self):
        super().__init__()

    def audio_files_list(self, root, ext=['wav'], recurse=True):
        files = librosa.util.find_files(root, ext=ext, recurse=recurse)

        print('Found {} files in {}\n'.format(len(files), root) +
              'Audio to spectogram images process has started...')
        return files
    
    def blocks_num(self, file):
        # TODO -> take care of segment length and overlapping frames
        return '?'

    def to_mono(self, data):
        # 1D -> Already mono
        if len(data.shape) == 1:
            return data
        # 2D -> Stereo to mono
        elif len(data.shape) == 2:
            return np.mean(data, axis=1)
        raise TypeError('Audio data must be mono or stereo. More channels are not supported!')
    
    def to_spectrogram(self, y):
        ''' Transform time-series to spectrogram '''
        #stft_matrix = librosa.stft(y)
        #magnitute_matrix = np.abs(stft_matrix)**2
        #mel_spectrogram = librosa.feature.melspectrogram(S=magnitute_matrix, sr=self.sr, fmax=self.max_frequency)


        # TODO wrap this nicely: n_mels is adjusted for MobileNet
        mel_spectrogram = librosa.feature.melspectrogram(y, n_mels=224)
        return librosa.power_to_db(mel_spectrogram)

    @utils.print_execution_time
    def transform_to_spectogram_segments(self, input_dir, output_dir):
        for file_path in self.audio_files_list(input_dir):
            file = File(file_path)
            
            # Divide audio into blocks
            print('Processing {}...\n'.format(file.full_name))
            blocks = sf.blocks(file_path, 
                            blocksize=int(file.original_sampling_rate * self.segment_duration),
                            overlap=int(file.original_sampling_rate * self.segment_overlap),
                                dtype=self.dtype)
            # Process each block
            for block_idx, block_data in enumerate(blocks):
                start_time = time.clock()

                y = self.to_mono(block_data)
                y = librosa.resample(y, file.original_sampling_rate, self.sr)

                # Classify very silent blocks as empty -> won't generate a spectrogram
                if (y < self.silence_threshold).all():
                    print("[✗] {}/{} block contains silence only, omitting spectrogram generation process."
                          .format(block_idx, self.blocks_num(file)))
                    continue

                # Output to file as an image
                image.imsave(file.output_path(block_idx, output_dir), self.to_spectrogram(y))

                # Store in binary format
                #np.save(file.output_path(block_idx, output_dir), self.to_spectrogram(y)) 

                # Display updates to the console
                millis = int(round((time.clock() - start_time) * 1000))
                print('[✓] {}/{} took {}ms'.format(block_idx + 1, self.blocks_num(file), millis))
            print('DONE ✓')

    @utils.print_execution_time
    def single(self, input_path, output_dir):
        file = File(input_path)

        # Divide audio into blocks
        print('Processing {}...\n'.format(file.full_name))
        blocks = sf.blocks(input_path,
                            blocksize=int(
                                file.original_sampling_rate * self.segment_duration),
                            overlap=int(
                                file.original_sampling_rate * self.segment_overlap),
                            dtype=self.dtype)
        # Process each block
        for block_idx, block_data in enumerate(blocks):
            start_time = time.clock()

            y = self.to_mono(block_data)
            y = librosa.resample(y, file.original_sampling_rate, self.sr)

            # Classify very silent blocks as empty -> won't generate a spectrogram
            if (y < self.silence_threshold).all():
                print("[✗] {}/{} block contains silence only, omitting spectrogram generation process."
                        .format(block_idx, self.blocks_num(file)))
                continue

            # Output to file as an image
            image.imsave(file.output_path(
                block_idx, output_dir, without_folder_structure=True), self.to_spectrogram(y))

            # Store in binary format
            #np.save(file.output_path(block_idx, output_dir), self.to_spectrogram(y))

            # Display updates to the console
            millis = int(round((time.clock() - start_time) * 1000))
            print('[✓] {}/{} took {}ms'.format(block_idx +
                                                1, self.blocks_num(file), millis))
        print('DONE ✓')


if __name__ == '__main__':
    '''Preprocessor assumes that the dataset has structure like following:
        
        ├── train
        |   ├── cello
        |   ├── ...
        |   ├── ...
        |   └── piano
        |
        └── test
            ├── cello
            ├── ...
            ├── ...
            └── piano
    '''
    parser = argparse.ArgumentParser(
        description='Dataset preprocessor for instrument recognition task with DNN')

    parser.add_argument('-i', '--input-dataset-dir',
                        default='/home/miczi/datasets/piano_and_cello',
                        action=utils.FullPaths,
                        type=utils.is_dir,
                        required=False,
                        help='Path to folder with dataset structure containing audio files \n \
                                (put into label-like folder name, e.g. barking.wav inside dogs/, miau.wav inside cats/)')

    parser.add_argument('-s', '--single-file-input')

    parser.add_argument('-o', '--output-spectrograms-dir',
                        default='/home/miczi/Projects/single-instrument-recognizer/output/spectrograms',
                        action=utils.FullPaths,
                        type=utils.is_dir,
                        required=False,
                        help='Path to destination folder for generated spectrograms')

    args = parser.parse_args()
    utils.print_parameters(args)

    # Change input and output paths in default_settings.py
    processor = Preprocessor()

    if args.single_file_input:
        processor.single(input_path=args.single_file_input, output_dir=args.output_spectrograms_dir)
    else:
        processor.transform_to_spectogram_segments(input_dir=args.input_dataset_dir, output_dir=args.output_spectrograms_dir)
