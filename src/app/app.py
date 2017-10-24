from flask import Flask, render_template, request, session, redirect, url_for, escape, request
from werkzeug import secure_filename
from flask_recaptcha import ReCaptcha

app = Flask(__name__)

#################
# Configuration
#################

# reCAPTCHA tokens
app.config.update({'RECAPTCHA_ENABLED': True,
                   'RECAPTCHA_SITE_KEY': '6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI',
                   'RECAPTCHA_SECRET_KEY': '6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe'})
recaptcha = ReCaptcha(app=app)

# TODO Very urgent: replace the string below with generated secret string
# and move it to separate file (make sure it's not tracked by git!)
# Example: NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
app.secret_key = 'InstrumentyDNN'

#################
# Routes
#################

# Home
@app.route('/')
def index():
    if 'username' in session:
        username = session['username']
        return render_template('index.html')
    return render_template('index.html')

# TODO Sign in (not working yet)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    return render_template('login.html')

# TODO Sign out (not working yet)
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if recaptcha.verify():
            f = request.files['file']
            f.save(secure_filename(f.filename))
            return 'file uploaded successfully'
        return 'wrong recaptcha'


if __name__ == '__main__':
    app.run(debug=True)
