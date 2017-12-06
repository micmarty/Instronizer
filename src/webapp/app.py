"""Flask webapp which allows to identify instrument present on a given audio file

1. Example usage:
- Production:
    (need to install uwsgi first: pip install uwsgi)
    uwsgi --socket 127.0.0.1:5000 --protocol=http --wsgi-file <PATH_TO_THIS_FILE> --callable app --processes 4 --threads 2 --stats 127.0.0.1:9191 \
    --pyargv '--checkpoint <PATH_TO_CHECKPOINT>'

- Development:
    python src/webapp/app.py --checkpoint <PATH_TO_THIS_FILE> --port <SOME_AVAILABLE_PORT_NUMBER>

2. Args:
    --checkpoint <PATH_TO_THIS_FILE> (required)
    --port <SOME_AVAILABLE_PORT_NUMBER> (optional)

3. External dependencies:
ffmpeg-python - https://github.com/kkroening/ffmpeg-python
lightweight_classifier.py source file
"""

__copyright__ = 'Copyright 2017, Instronizer'
__credits__ = ['Michał Martyniak', 'Maciej Rutkowski', 'Filip Schodowski']
__license__ = 'MIT'
__version__ = '1.0.0'
__status__ = 'Production'


from flask import Flask, render_template, request, session, redirect, url_for, escape, Response, jsonify
from werkzeug import secure_filename
from pathlib import Path
from argparse import ArgumentParser
import subprocess
import ffmpeg
import time
import arrow
import uwsgi
from classifier.utils.printing_functions import print_execution_time

# Relative to application application source root
from webapp import lightweight_classifier

##
# Constants
CURRENT_DIR = Path(__file__).absolute().parent
AUDIO_DIR = Path('./upload/wav')
TMP_DIR = Path('./upload/tmp')
SPECS_DIR = Path('./upload/specs')
PROCESSOR_PATH = CURRENT_DIR.parent / 'preprocessor/wav_to_spectrograms.py'
ALLOWED_EXTENSIONS = ['.wav', '.flac', '.mp3']
MAX_UPLOAD_SIZE = 50  # MB
SPEC_LENGTH = 3  # in seconds
UPLOADS_DELETION_TIME = 5  # in minutes

##
# Print app info to the console
print('\n=== APP INFO ===')
print('CURRENT_DIR={}\n\nAUDIO_DIR={}\nSPECS_DIR={}\nPREPROCESSOR_PATH={}\nSPEC_LENGTH={}\n'.format(
    CURRENT_DIR, AUDIO_DIR, SPECS_DIR, PROCESSOR_PATH, SPEC_LENGTH))
print('ALLOWED_EXTENSIONS={}'.format(ALLOWED_EXTENSIONS))
print('================\n')

##
# Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = AUDIO_DIR
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_SIZE * 1024 * 1024
app.secret_key = 'TODO use random value'

##
# Functions
def input_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default='', type=str, required=True,
                        metavar='PATH', help='File with saved weights (some state of trained model)')
    parser.add_argument('-p', '--port', default=5000, type=int, metavar='PORT')
    return parser.parse_args()


def delete_files(folder_path):
    criticalTime = arrow.now().shift(minutes=-UPLOADS_DELETION_TIME)
    for element in folder_path.iterdir():
        if element.is_dir():
            delete_files(element)
            if list(element.iterdir()) == []:
                element.rmdir()
        else:
            elementTime = arrow.get(element.stat().st_atime)
            if elementTime < criticalTime:
                print(element)
                element.unlink()


def delete_unused_files(signum):
    """Function called by uwsgi periodically
    Removes unused files, keeps the disk clean
    """
    print("Removing unused files...")
    delete_files(AUDIO_DIR)
    delete_files(TMP_DIR)
    delete_files(SPECS_DIR)


uwsgi.register_signal(1, '', delete_unused_files)
uwsgi.add_timer(1, UPLOADS_DELETION_TIME * 60) # conversion to seconds


@print_execution_time
def generate_spectrograms(audio_filename, time_range, length, offset):
    """Transforms wav file excerpt into a spetrogram.
    It simply executes python script inside a shell.

    Returns:
        A preprocessor script's exit code
    """
    command = 'python {script_path} --input {audio_path} --output-dir {spec_dir} --start {start} --end {end} --segment-length {length} --segment-overlap-length {overlap}'.format(
        script_path=PROCESSOR_PATH,
        audio_path=AUDIO_DIR / audio_filename,
        spec_dir=SPECS_DIR,
        start=time_range[0],
        end=time_range[1],
        length=length,
        overlap=offset
    )
    exit_code = subprocess.check_call(command, shell=True)
    spectrograms_dir = SPECS_DIR / Path(audio_filename).stem
    return exit_code, spectrograms_dir


@print_execution_time
def classify(spectrograms_dir):
    """Launches a lighweight_classifier script

    Args:
        spectrogram_dir: string, path pointing to existing dir, containging .npy spectrograms
        checkpoint_path: string, path to saved, trained model checkpoint
    Returns:
        A list of floats (output vector from model)
    """
    global args
    return lightweight_classifier.run(spectrograms_dir, checkpoint_path=args.checkpoint)


@print_execution_time
def convert_to_wav(tmp_path, dest_path):
    """Convert between any audio format supported by ffmpeg

    Args:
        tmp_path: string, input file path with some audio extension at the end
                    (example: ./data/foo/name.mp3)
        dest_path: string, output path to which file will be converted to 
                    (example: ./data/bar/name.wav)
    """
    stream = ffmpeg.input(tmp_path)
    stream = ffmpeg.output(stream, dest_path)
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)

##
# Routes
@app.route('/')
def index():
    return render_template('layout.html', max_upload=MAX_UPLOAD_SIZE)


@app.route('/upload', methods=['POST'])
def upload():
    # Parse request
    file = request.files['file']

    # Check if file is valid
    if file and Path(file.filename).suffix in ALLOWED_EXTENSIONS:
        # Prevent path attack (e.g. ../../../../somefile.sh) and extract essential info
        filename = secure_filename(file.filename)
        basename = Path(filename).stem
        extenstion = Path(filename).suffix
        mimetype = file.content_type

        # Prepare paths
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path = str(TMP_DIR / filename)
        destination_path = str(AUDIO_DIR / (basename + '.wav'))

        # Convert supported formats to wav
        file.save(tmp_path)
        convert_to_wav(tmp_path, destination_path)

        # Send back to client
        return jsonify(success=True, path=str(basename + '.wav'))


@app.route('/classify', methods=['POST'])
def get_instruments():
    file_path = request.form['file_path']
    start = round(float(request.form['start']))
    end = round(float(request.form['end']))

    # Run preprocessing and classification on trained neural network
    exit_code, spectrograms_dir = generate_spectrograms(
        file_path, time_range=(start, end), length=SPEC_LENGTH, offset=SPEC_LENGTH)

    # If success
    if exit_code == 0:
        instruments_results_list = classify(spectrograms_dir)
        return render_template('results.html', start=start, end=end, result=instruments_results_list)
    return jsonify(start=start, end=end, result='PREPROCESSOR_ERROR')


args = input_args()
if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=args.port)
