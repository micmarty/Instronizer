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
