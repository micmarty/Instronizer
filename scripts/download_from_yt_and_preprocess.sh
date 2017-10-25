#!/bin/bash
###
# This script downloads yt videos, extracts audio and runs preprocessing (wav to spectrogram)
#
# It requires to run from project root!
#
# It takes two arguments (both should be full paths): 
# <path to formatted file containing YT links> and <output folder for wav and spectrograms>

INPUT_FILE_PATH=$1
OUTPUT_DIR=$2

# Parse and download audio from input file containing links
python src/utils/yt_downloader.py --input-file $INPUT_FILE_PATH --output-dir $OUTPUT_DIR

for file in $OUTPUT_DIR/wav/*
do
    # Get file name without the extension
    filename=$(basename "$file" .wav)

    # Normalize the loudness
    ffmpeg-normalize --no-prefix --force $OUTPUT_DIR/wav/*.wav 

    # Prepare separate folder for each wav file
    mkdir --parents $OUTPUT_DIR/spectrograms/$filename

    # Run the preprocessor
    python src/preprocessor.py --single-file-input $file --output-spectrograms-dir $OUTPUT_DIR/spectrograms/$filename
done