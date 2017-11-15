from flask import Flask, render_template, request, session, redirect, url_for, escape, Response,jsonify
from werkzeug import secure_filename
from pathlib import Path
import subprocess
from classifier.utils.printing_functions import print_execution_time
# Relative to application application source root
from webapp import lightweight_classifier

##
# Constants
CURRENT_DIR = Path(__file__).absolute().parent
AUDIO_DIR = Path('./data/audio')
SPECS_DIR = Path('./data/specs')
PROCESSOR_PATH = CURRENT_DIR.parent / 'preprocessor/wav_to_spectrograms.py'
ALLOWED_EXTENSIONS = ['.wav']

print('\n=== APP INFO ===')
print('CURRENT_DIR={}\n\nAUDIO_DIR={}\nSPECS_DIR={}\nPREPROCESSOR_PATH={}\n'.format(
    CURRENT_DIR, AUDIO_DIR, SPECS_DIR, PROCESSOR_PATH))
print('ALLOWED_EXTENSIONS={}'.format(ALLOWED_EXTENSIONS))
print('================\n')

##
# Configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = AUDIO_DIR
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 60 MB
app.secret_key = 'TODO use random value'

##
# Functions


@print_execution_time
def generate_spectrograms(audio_filename, time_range):
    '''
    Transforms wav into spetrograms
    Returns exit code for preprocessing operation
    '''
    command = 'python {script_path} --input {audio_path} --output-dir {spec_dir} --start {start} --end {end}'.format(
        script_path=PROCESSOR_PATH,
        audio_path=AUDIO_DIR / audio_filename,
        spec_dir=SPECS_DIR,
        start=time_range[0],
        end=time_range[1]
    )
    exit_code = subprocess.check_call(command, shell=True)
    spectrograms_dir = SPECS_DIR / Path(audio_filename).stem
    return exit_code, spectrograms_dir


@print_execution_time
def classify(spectrograms_dir):
    '''
    Runs simplified classificator
    Returns list of floats
    '''
    return lightweight_classifier.run(spectrograms_dir)

##
# Routing
@app.route('/')
def index():
    return render_template('layout.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Parse request
    file = request.files['file']
    
    # Check if file is valid
    if file and Path(file.filename).suffix in ALLOWED_EXTENSIONS:
        # Prevent path attack (e.g. ../../../../somefile.sh)
        filename = secure_filename(file.filename)

        # Make sure folder for uploaded audio files exists
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)

        # Build destination path and save
        destination_path = str(AUDIO_DIR / filename)
        file.save(destination_path)

        # Send back to client
        return jsonify(success=True, path=str(filename))

@app.route('/classify', methods=['POST'])
def get_instruments():
    file_path = request.form['file_path']
    start = round(float(request.form['start']))
    end = round(float(request.form['end']))

    # Run preprocessing and classification on trained neural network
    exit_code, spectrograms_dir = generate_spectrograms(
        file_path, time_range=(start, end))
    if exit_code == 0:
        instruments_results_list = classify(spectrograms_dir)
        return render_template('results.html', start=start, end=end, result=instruments_results_list)
    return jsonify(start=start, end=end, result='PREPROCESSOR_ERROR')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
