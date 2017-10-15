import easygui
from pathlib import Path, PurePath
import subprocess

file_path = easygui.fileopenbox(
    default='~/Desktop/*.wav',
    filetypes=['*.wav'],
    title='Choose a single wav file with solo instrument')

out_dir = easygui.diropenbox(
    default='~/Music',
    title='Choose ouput directory for generated spectrograms')

# Create output dir
file_path = Path(file_path) 
out_dir = Path(out_dir, 'spectrograms')
out_dir.mkdir(exist_ok=True)
print("Created directory: ", out_dir)

# Execute preprocessor
subprocess.check_call(['python', '/home/miczi/Projects/single-instrument-recognizer/src/preprocessor.py', 
                        '--single-file-input', str(file_path), 
                        '--output-spectrograms-dir', str(out_dir)])

subprocess.check_call(['python', '/home/miczi/Projects/single-instrument-recognizer/src/mobilenet_train.py',
                        '-a', 'mobilenet',
                        str(out_dir),
                        '--print-freq', '1',
                        '--resume', '/home/miczi/Desktop/mobilenet_epoch_3/checkpoint.pth.tar',
                        '--classify-spectrograms'])
