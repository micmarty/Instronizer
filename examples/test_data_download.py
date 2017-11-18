import subprocess
from pathlib import Path

DESTINATION_DIR = Path.cwd() / 'test'


test_data = {
    'cello': [
        {
            'title': 'alban_gerhardt',
            'link': 'https://www.youtube.com/watch?v=fP_RnCUlLHg',
            'excerpts': [
                ('0:00:18', '0:04:25'), ('0:04:32', '0:07:58')
            ]
        }
    ]
}

youtube_dl_command = 'youtube-dl --extract-audio --audio-format wav --output {output_path} {link}'
ffmpeg_cut_command = 'ffmpeg -i {input_path} -ss {start} -t {end} {output_path}'

def execute_string(command):
    '''Returns command exit code'''
    # Call requires each argument to be a separate list element
    command = command.split(' ')
    return subprocess.check_call(command, shell=False)

def dump_excerpt(source_path, excerpt_path, start, end):
    '''
    Extracts a single excerpt from file with start and end parameters given
    Returns exit code
    '''
    # FFmpeg cut excerpt
    command = ffmpeg_cut_command.format(input_path=source_path, output_path=excerpt_path, start=start, end=end)
    return execute_string(command)
    
def download_sample(link, output_path):
    '''
    Downloads a YouTube video as wav audio
    Returns exit code
    '''
    # Download whole wav file
    command = youtube_dl_command.format(output_path=sample_path, link=sample['link'])
    return execute_string(command)


for instrument in test_data:
    # Create dir for instrument
    instrument_dir = DESTINATION_DIR / instrument
    instrument_dir.mkdir(parents=True, exist_ok=True)

    for sample in test_data[instrument]:
        sample_path = DESTINATION_DIR / '{basename}.wav'.format(basename=sample['title'])
        download_sample(link=sample['link'], output_path=sample_path)

        # Divide downloaded wav file into excerpts with unique names
        for time_range in sample['excerpts']:
            start, end = time_range
            excerpt_path = instrument_dir / '{name}_{start}-{end}.wav'.format(name=sample['title'], start=start, end=end)
            dump_excerpt(sample_path, excerpt_path, start, end)
            print('Successfully downloaded: {} - ({}, {})'.format(sample['title'], start, end))


