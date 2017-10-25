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
    return render_template('v2/index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if recaptcha.verify():
            f = request.files['file']
            f.save(secure_filename(f.filename))
            return 'file uploaded successfully'
        return 'wrong recaptcha'


if __name__ == '__main__':
    app.run(debug=True)

