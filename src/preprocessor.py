import librosa
import numpy as np
import soundfile as sf
import matplotlib
import matplotlib.image as image
import matplotlib.pyplot as plt
import os

# TODO omit the last spectrogram segment because it's always shorter than the others
from utils import count_elapsed_time



class AudioToImageProcessor():
    # CONSTANTS for fast-switching
    SAMPLING_RATE = 22050
    SEGMENT_DURATION_IN_S = 5.9 # this gives 128px width on spectrogram (empirically selected)
    SEGMENT_OVERLAP_IN_S = 2.5

    def __init__(self, input_folder, output_folder):
        # Folder locations
        self.root = input_folder
        self.output_folder = output_folder

        # Processing parameters
        self.sampling_rate = AudioToImageProcessor.SAMPLING_RATE
        self.segment_frames_num = int(self.sampling_rate * AudioToImageProcessor.SEGMENT_DURATION_IN_S) 
        self.overlapping_frames_num = int(self.sampling_rate * AudioToImageProcessor.SEGMENT_OVERLAP_IN_S)

        # Other parameters
        self.dtype = 'float32'
        self.max_frequency = self.sampling_rate / 2                     # librosa's default

    @count_elapsed_time
    def transform_to_spectogram_segments(self):
        wav_files = librosa.util.find_files(self.root, ext=['wav'], recurse=True)
        print('Found {} files in {}\n'.format(len(wav_files), self.root) +
                'Output folder: {}\n'.format(self.output_folder) + 
                'Audio to spectogram images processing has started...')

        for file in wav_files:
            # 1. Extract basic info from file
            file_info = sf.info(file)
            original_filename = os.path.basename(file_info.name)       # truncate whole path and leave only basename
            original_filename = os.path.splitext(original_filename)[0] # truncate the extension
            original_sampling_rate = file_info.samplerate

            # 2. Split audio into blocks
            blocks_num = int(file_info.frames / self.segment_frames_num * (original_sampling_rate / self.sampling_rate))
            
            print('Processing {}...\n'.format(original_filename), end=' ')
            blocks = sf.blocks(file, blocksize=self.segment_frames_num, 
                                overlap=self.overlapping_frames_num, 
                                dtype=self.dtype)

            # 3. Process blocks one by one
            for block_id, block_data in enumerate(blocks):
                # 3.1 Downmix stereo to mono
                if len(block_data.shape) == 1:
                    # 1D -> Already mono
                    y = block_data
                if len(block_data.shape) == 2:
                    # 2D -> stereo to mono
                    y = np.mean(block_data, axis=1)

                # 3.2 Downsample
                y = librosa.resample(y, original_sampling_rate, self.sampling_rate)

                # 3.3 Classify very silent blocks as empty -> won't generate a spectrogram
                if (y < 0.02).all():
                    print("{}/{} block contains silence only, omitting spectrogram generation process."
                        .format(block_id, blocks_num))
                    continue

                # TODO this seems to be useless -> we need to have fixed image sizes!
                # 3.4 Trim silence from beginning and end
                # y, _ = librosa.effects.trim(y) 

                ##
                # From librosa docs: https://librosa.github.io/librosa/generated/librosa.core.stft.html
                # librosa.stft(y) -> Returns a complex-valued matrix D such that
                #   np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
                #   np.angle(D[f, t]) is the phase of frequency bin f at frame t
                #
                # 3.5 STFT and spectrogram
                stft_matrix = librosa.stft(y)
                magnitute_matrix = np.abs(stft_matrix)**2
                mel_spectrogram = librosa.feature.melspectrogram(S=magnitute_matrix, sr=self.sampling_rate, fmax=self.max_frequency)

                # 3.6 Prepare path for saving the spectrogram

                # e.g. 'cello', 'piano'
                label_dirname = os.path.basename(os.path.dirname(file_info.name))
                output_segment_name = '{}_{}.png'.format(original_filename, block_id + 1)
                output_folder = os.path.join(self.output_folder, label_dirname)

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                output_path = os.path.join(output_folder, output_segment_name)

                # 3.7 Finally save the spectrogram
                image.imsave(output_path, librosa.power_to_db(mel_spectrogram))

                print('{}/{}'.format(block_id+1, blocks_num))
            print('DONE')


processor = AudioToImageProcessor(input_folder='/home/miczi/datasets/piano_and_cello', 
                                output_folder='/home/miczi/Projects/single-instrument-recognizer/output/spectrograms')

processor.transform_to_spectogram_segments()
