FROM tiangolo/uwsgi-nginx-flask:python3.6

RUN apt-get update && apt-get -y install ffmpeg
RUN pip install --upgrade youtube_dl
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl torchvision
RUN pip install ffmpeg-python \
    arrow \
    uwsgi \
    librosa==0.5.1 \
    soundfile \
    scipy \
    better_exceptions

# pip install in case of compatibility issues in the future
# arrow==0.12.0
# uwsgi==2.0.15
# librosa==0.5.1
# soundfile==0.9.0p
# scipy==1.0.0

# $ pip freeze
# arrow==0.12.1
# audioread==2.1.5
# better-exceptions==0.2.1
# cffi==1.11.5
# click==6.7
# decorator==4.2.1
# ffmpeg-python==0.1.10
# Flask==0.12.2
# future==0.16.0
# itsdangerous==0.24
# Jinja2==2.10
# joblib==0.11
# librosa==0.5.1
# llvmlite==0.22.0
# MarkupSafe==1.0
# numba==0.37.0
# numpy==1.14.1
# Pillow==5.0.0
# pycparser==2.18
# python-dateutil==2.6.1
# PyYAML==3.12
# resampy==0.2.0
# scikit-learn==0.19.1
# scipy==1.0.0
# six==1.11.0
# SoundFile==0.10.1
# torch==0.3.0.post4
# torchvision==0.2.0
# uWSGI==2.0.15
# Werkzeug==0.14.1
# youtube-dl==2018.3.3

ENV NGINX_MAX_UPLOAD 50m
ENV LISTEN_PORT 80
ENV UWSGI_INI /app/src/webapp/uwsgi.ini
ENV STATIC_URL /static
ENV STATIC_PATH /app/src/webapp/static
# If STATIC_INDEX is 1, serve / with /static/index.html directly (or the static URL configured) 
# ENV STATIC_INDEX 1 
ENV STATIC_INDEX 0

COPY . /app
WORKDIR /app/src/webapp
ENV PYTHONPATH=/app/src
