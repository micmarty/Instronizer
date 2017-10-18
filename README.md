## Current progress

Using IRMAS dataset and manually collected excerpts from YouTube

## Open mobilenet_train.py for code which is currently used
```bash
# see required folder structure inside preprocessor.py
# output dir must exists earlier
python src/preprocessor.py -i <wav_dataset_dir> -o <spectrogram_output_dir>

# before use, set constants inside the file
python src/list_images_with_wrong_dim.py

# enable tensorboard
# make sure that 'logs' folder is empty before starting the new training (otherwise it will concatenate old and new data)
tensorboard --logdir='./logs' --port=6006

# start the training
python src/mobilenet_train.py -a mobilenet <path_to_spectrograms_224x224>

# open browser at: localhost:6006
```

## Useful commands on remote training server
```bash
oi  # this alias opens project directory, all further instructions are there
# # On remote -> tab 1

# # For the first time
# cd <projects>
# git clone https://github.com/micmarty/instrument-classifier-polyphonic.git

# # Each next time
# git status
# git pull

# source venv/pytorch_env/bin/activate
# jupyter notebook --no-browser --port 8889

# # On remote -> tab 2
# watch --interval 0 nvidia-smi

# # On local machine
# ssh -NfL localhost:8888:localhost:8889 <user>@<server>
# # Launch your browser at localhost:8888
```

**Accuracy**:

(update soon)

- validation set - 11 instruments

![Validation](https://image.ibb.co/ig3rRG/Pasted_image_at_2017_10_11_08_56_PM.png)

- test set - 3/11 instruments (sax, cello, piano)

![Testing](https://image.ibb.co/kWZLLb/Pasted_image_at_2017_10_11_09_28_PM.png)

## Docs will be provided at some reasonable developement stage


