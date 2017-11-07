import os
import time
import subprocess
from flask import Flask, render_template, request, session, redirect, url_for, escape
from werkzeug import secure_filename
from flask_recaptcha import ReCaptcha


UPLOAD_FOLDER = './output/upload/'
SPECTROGRAMS_FOLDER = './output/spectrograms/'
ALLOWED_EXTENSIONS = set(['mp3', 'wav'])

app = Flask(__name__)

#################
# Configuration
#################

if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    if not os.path.isdir(SPECTROGRAMS_FOLDER):
        os.makedirs(SPECTROGRAMS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024 # max 20 mb
# reCAPTCHA tokens
app.config.update({'RECAPTCHA_ENABLED': True,
                   'RECAPTCHA_SITE_KEY': '6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI',
                   'RECAPTCHA_SECRET_KEY': '6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe'})
recaptcha = ReCaptcha(app=app)

# TODO Very urgent: replace the string below with generated secret string
# and move it to separate file (make sure it's not tracked by git!)
# Example: NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
app.secret_key = 'InstrumentyDNN'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def create_spectrogram(filename):
    SCRIPT_PATH = os.getcwd() + '/src/classifier/wav_to_spectrograms.py'
    FILE_PATH = UPLOAD_FOLDER + '/' + filename
    subprocess.check_call(['python', SCRIPT_PATH, '-i', FILE_PATH, '-o', SPECTROGRAMS_FOLDER])

#################
# Routes
#################

# Home
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if recaptcha.verify():
            file = request.files['file']
            start = int(float(request.form['start']))
            end = int(float(request.form['end']))
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                create_spectrogram(filename)
    return render_template('uploading.html', sof=start, eof=end)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
