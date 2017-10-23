from flask import Flask, render_template, request, session, redirect, url_for, escape, request
from werkzeug import secure_filename
from flask_recaptcha import ReCaptcha

app = Flask(__name__)
#konfiguracja recaptcha
app.config.update({'RECAPTCHA_ENABLED': True,
                   'RECAPTCHA_SITE_KEY':
                       '6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI',
                   'RECAPTCHA_SECRET_KEY':
                       '6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe'})
recaptcha = ReCaptcha(app=app)

app.secret_key = 'InstrumentyDNN'

#sciezka do strony glownej
@app.route('/')
def index():
   if 'username' in session:
      username = session['username']
      return render_template('index.html')
   return render_template('index.html')

#sciezka do strony logowania (jeszcze nie dziala)
@app.route('/login', methods = ['GET', 'POST'])
def login():
   if request.method == 'POST':
      session['username'] = request.form['username']
      return redirect(url_for('index'))
   return render_template('login.html')

#wylogowanie (jeszcze nie dziala)
@app.route('/logout')
def logout():
   session.pop('username', None)
   return redirect(url_for('index'))

#sciezka do strony uploadu
@app.route('/upload')
def upload_file():
   return render_template('upload.html')

#upload
@app.route('/uploader', methods = ['GET', 'POST'])
def upload():
   if request.method == 'POST':
      if recaptcha.verify():
        # SUCCESS
         f = request.files['file']
         f.save(secure_filename(f.filename))
         return 'file uploaded successfully'
         pass
      else:
        # FAILED
         return 'wrong recaptcha'
         pass
      
		
if __name__ == '__main__':
   app.run(debug = True)
