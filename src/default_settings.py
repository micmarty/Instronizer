PATHS = dict(
    INPUT_DATASET_DIR = '/home/miczi/datasets/piano_and_cello',
    OUTPUT_SPECTROGRAM_DIR = '/home/miczi/Projects/single-instrument-recognizer/output/spectrograms'
)

HELP = dict(
    INPUT_DATASET_DIR='Path to folder with dataset structure containing audio files \n \
                            (put into label-like folder name, e.g. barking.wav inside dogs/, miau.wav inside cats/)',
    OUTPUT_SPECTROGRAM_DIR='Path to destination folder for generated spectrograms'
)

STRINGS = dict(
    ARG_PARSER_DESCRIPTION='Instrument recognition with DNN in PyTorch'
)
