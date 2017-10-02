import librosa
import numpy as np
import soundfile as sf
import matplotlib
import matplotlib.image as image
import os

# TODO omit the last spectrogram segment because it's always shorter than the others



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

    def transform_to_spectogram_segments(self):
        wav_files = librosa.util.find_files(self.root, ext=['wav'], recurse=False)
        print('Found {} files in {}\n'.format(len(wav_files), self.root) +
                'Output folder: {}\n'.format(self.output_folder) + 
                'Audio to spectogram images processing has started...')

        for file in wav_files:
            # Extract basic info from file
            file_info = sf.info(file)
            original_filename = os.path.basename(file_info.name)       # truncate whole path and leave only basename
            original_filename = os.path.splitext(original_filename)[0] # truncate the extension
            original_sampling_rate = file_info.samplerate

            # TODO fix bad results
            blocks_num = int(file_info.frames / self.segment_frames_num * (original_sampling_rate / self.sampling_rate))
            
            print('Processing {}...\n'.format(original_filename), end=' ')
            blocks = sf.blocks(file, blocksize=self.segment_frames_num, 
                                overlap=self.overlapping_frames_num, 
                                dtype=self.dtype)

            # Process chunks one by one
            for block_id, block_data in enumerate(blocks):
                # TODO make sure that mono input won't crash

                # Downmix stereo to mono and 
                if len(block_data.shape) == 1:
                    # 1D -> Already mono
                    y = block_data
                if len(block_data.shape) == 2:
                    # 2D -> stereo to mono
                    y = np.mean(block_data, axis=1)

                # Downsample
                y = librosa.resample(y, original_sampling_rate, self.sampling_rate)

                if (y<0.02).all():
                    print("{}/{} block contains silence only, omitting spectrogram generation process."
                        .format(block_id, blocks_num))
                    continue

                # Trim silence from beginning and end
                y, _ = librosa.effects.trim(y) 
                ##
                # From librosa docs: https://librosa.github.io/librosa/generated/librosa.core.stft.html
                # librosa.stft(y) -> Returns a complex-valued matrix D such that
                #   np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
                #   np.angle(D[f, t]) is the phase of frequency bin f at frame t
                #
                stft_matrix = librosa.stft(y)
                magnitute_matrix = np.abs(stft_matrix)**2
                
                # TODO try to generate mel spec from y, sr and without fmax
                mel_spectrogram = librosa.feature.melspectrogram(S=magnitute_matrix, sr=self.sampling_rate, fmax=self.max_frequency)

                # Output file preparation
                if not os.path.exists(self.output_folder):
                    os.makedirs(self.output_folder)

                output_segment_name = '{}_{}.png'.format(original_filename, block_id+1)
                output_path = os.path.join(self.output_folder, output_segment_name)
                
                image.imsave(output_path, librosa.power_to_db(mel_spectrogram))

                print('{}/{}'.format(block_id+1, blocks_num))
            print('DONE')


processor = AudioToImageProcessor(input_folder='/home/miczi/datasets/piano_and_cello', 
                                output_folder='/home/miczi/Projects/single-instrument-recognizer/output/spectrograms')

processor.transform_to_spectogram_segments()
