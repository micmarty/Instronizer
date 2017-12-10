# Instronizer

## Description

**Instronizer** - instrument recognition system based on a usage of DNN (Deep Neural Networks)

At this point it was trained to identify 6 musical instruments:
- Cello
- Acoustic guitar
- Electric guitar
- Church organs
- Piano
- Violin

## Accuracy

It performs pretty well, achieving **86%** on our test dataset and **91%** on validation dataset.
At the beginning we were using **IRMAS** dataset but after some trainings and checking the labels, it turned out to be "not well made".
For this reason we've built our own dataset from YouTube.

This project was designed, written and is maintained by **Michał Martyniak**, **Maciej Rutkowski**, **Filip Schodowski**

## System components

- Web application (Python, Flask, uWSGI, nginx, Travis CI, Docker, Google Material Design)

- Convolutional Neural Network (PyTorch framework, MobileNet architecture)

- Data preprocessor (Python) - WAV -> normalization -> downmixing -> downsampling -> mel-scaled spectrograms -> many .npy files

- IRMAS dataset:
    - 10 instruments + voice
    - very unbalanced
    - different labeling structure for train and test sets
    - very poor quality of annotations (wrong labels was a pain)
    - 20 - 39 min per instrument
    - Many excerpts from one song

- YouTube dataset: 
    - 6 instruments
    - Handmade by us
    - train (2h per instrument)
    - val (30 min per instrument)
    - test (30 min per instrument, about 1 min limit for every YouTube audio clip - diversified)

- Auxiliary scripts: YouTube downloader and parser


## Play with the code

You need to set **PYTHONPATH** environment variable first.
Code uses imports relative to ```<project_path>/src```
**In case of any problem, please contact us or leave a pull request**
```bash
cd <project_path>
# It's essential to import modules properly
export PYTHONPATH=<project_path>/src
```

## Web application demo - instant and easy to deploy with Docker image

```bash
cd <project_path>
docker build . --tag instronizer
docker run -p 80:80 --name instronizer_container instronizer
```
