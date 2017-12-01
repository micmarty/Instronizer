from flask import Flask, render_template, request, session, redirect, url_for, escape, Response,jsonify
from werkzeug import secure_filename
from pathlib import Path
from argparse import ArgumentParser
import subprocess
import ffmpeg
from classifier.utils.printing_functions import print_execution_time

# Relative to application application source root
from webapp import lightweight_classifier

##
# Constants
CURRENT_DIR = Path(__file__).absolute().parent
AUDIO_DIR = Path('./data/audio')
NON_WAV_DIR = Path('./data/non_wav_audio')
SPECS_DIR = Path('./data/specs')
PROCESSOR_PATH = CURRENT_DIR.parent / 'preprocessor/wav_to_spectrograms.py'
ALLOWED_EXTENSIONS = ['.wav', '.flac', '.mp3']
MAX_UPLOAD_SIZE = 200 # MB
SPEC_LENGTH = 3 # in seconds

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
    parser.add_argument('-c', '--checkpoint', default='', type=str, required=True, metavar='PATH', help='File with saved weights (some state of trained model)')
    parser.add_argument('-p', '--port', default=5000, type=int, metavar='PORT')
    return parser.parse_args()

@print_execution_time
def generate_spectrograms(audio_filename, time_range, length, offset):
    '''Transforms wav file excerpt into a spetrogram.
    It simply executes python script inside a shell.

    Returns:
        A preprocessor script's exit code
    '''
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
    '''Launches a lighweight_classifier script

    Args:
        spectrogram_dir: string, path pointing to existing dir, containging .npy spectrograms

    Returns:
        A list of floats (output vector from model)
    '''
    return lightweight_classifier.run(spectrograms_dir, checkpoint_path=args.checkpoint)

@print_execution_time
def convert_to_wav(tmp_path, dest_path):
    '''Convert between any audio format supported by ffmpeg
    
    Args:
        tmp_path: string, input file path with some audio extension at the end
                    (example: ./data/foo/name.mp3)
        dest_path: string, output path to which file will be converted to 
                    (example: ./data/bar/name.wav)
    '''
    stream = ffmpeg.input(tmp_path)
    stream = ffmpeg.output(stream, dest_path)
    stream = ffmpeg.overwrite_output(stream)
    ffmpeg.run(stream)

##
# Routes
@app.route('/')
def index():
    return render_template('layout.html')

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

        # Make sure folder for uploaded audio files exists
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        destination_path = str(AUDIO_DIR / (basename + '.wav'))
        
        # Uploaded file needs conversion to wav format
        if mimetype != 'audio/wav' and extenstion != '.wav':
            NON_WAV_DIR.mkdir(parents=True, exist_ok=True)
            # File with non-wav extension
            tmp_path = str(NON_WAV_DIR / filename)

            # Convert and save as wav
            file.save(tmp_path)
            convert_to_wav(tmp_path, destination_path)

        # Uploaded file is wav already
        else:
            file.save(destination_path)

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

if __name__ == '__main__':
    global args
    args = input_args()
    app.run(debug=True, threaded=True,port=args.port)
