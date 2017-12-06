"""Script for downloading test set from formatted text file containing YT video links

1. Example usage:
    python test_data_download.py -i <FORMATTED_TXT_FILE_PATH>

Example file:
    TODO

2. Args:
    -i, --input <PATH>

3. Required txt input file format:
```
[org]
1.
@get 1m26s-2m26s
https://www.youtube.com/watch?v=Ctykf8qh288 interstellar_first_step_zimmer
2.
@get 26s-1m26s
https://www.youtube.com/watch?v=UQDhH8-z14Q gigi_dagostino_lamour_toujours
3.
@get 4m36s-5m36s
https://www.youtube.com/watch?v=8nO6m7IkSK8 derrick_jackson_organ_solo
```

4. External dependencies:
youtube-dl - https://rg3.github.io/youtube-dl/
ffmpeg - https://www.ffmpeg.org/
"""

__copyright__ = 'Copyright 2017, Instronizer'
__credits__ = ['Michał Martyniak', 'Maciej Rutkowski', 'Filip Schodowski']
__license__ = 'MIT'
__version__ = '1.0.0'
__status__ = 'Production'

##
# Imports
import subprocess
import re
import datetime
from pathlib import Path
from argparse import ArgumentParser

##
# Constants and aliases
DESTINATION_DIR = Path.cwd() / 'test'
youtube_dl_command = 'youtube-dl --extract-audio --audio-format wav --output {output_path} {link}'
ffmpeg_cut_command = 'ffmpeg -i {input_path} -ss {start} -to {end} {output_path}'

def input_args():
    parser = ArgumentParser('Script for downloading test dataset')
    parser.add_argument('-i', '--input', required=True, type=str, metavar='<PATH>')
    return parser.parse_args()


def execute_string(command):
    """Returns command exit code"""
    # Call requires each argument to be a separate list element
    command = command.split(' ')
    return subprocess.check_call(command, shell=False)


def dump_excerpt(source_path, excerpt_path, start, end):
    """
    Extracts a single excerpt from file with start and end parameters given
    Returns exit code
    """
    # FFmpeg cut excerpt
    command = ffmpeg_cut_command.format(
        input_path=source_path, output_path=excerpt_path, start=start, end=end)
    return execute_string(command)


def download_sample(link, output_path):
    """
    Downloads a YouTube video as wav audio
    Returns exit code
    """
    # Download whole wav file
    command = youtube_dl_command.format(
        output_path=output_path, link=link)
    return execute_string(command)


def parse_excerpts_line(line):
    """
    Gets a string formatted like: '4m10s-5m25s, 1h15m2s-1h17m0s, ...'
    Returns list of tuples, like: [(0:04:10, 0:05:25), (1:15:2, 1:17:00), ...]
    """
    excerpts = []
    line = line.split(', ')
    for excerpt in line:
        # <start_time>-<end_time>
        # DANGER assuming there is only one '-'
        start_end_list = excerpt.split('-')
        start_end_tuple = ()
        for time in start_end_list:
            # Use non-capturing group '?:' to omit 'h', 'm', 's' in string
            # and extract only integer values
            h = '(?:([0-9]{1,2})h)*'
            m = '(?:([0-9]{1,2})m)*'
            s = '(?:([0-9]{1,2})s)'
            # Join into single regex expression
            regex = h + m + s
            h, m, s = re.match(regex, time).groups()

            # Take care of variables which are optional (example: h is None)
            h = h or 0
            m = m or 0
            # Convert into %H:%M:%S format
            formatted_time = str(datetime.timedelta(
                hours=int(h), minutes=int(m), seconds=int(s)))

            # Append time into tuple (up to 2 elements)
            start_end_tuple += (formatted_time, )
        excerpts.append(start_end_tuple)
        print(excerpts)
    return excerpts


def parse_file_to_dict(file_path):
    dictionary = {}
    current_instrument = ''
    current_excerpts = []
    with open(file_path) as f:
        for line in f:
            comment_blank_or_index = re.match(
                '[#]+.*', line) or re.match('^\s*$', line) or re.match('^\d+\.', line)
            instrument_name = re.match('^\[([a-z]{3})]', line)
            excerpt_line = re.match('^@get (.*)$', line)
            link_with_name = re.match('^(http.*) (.*)', line)
            # 1. Ignore comment or blank line
            if comment_blank_or_index:
                continue
            # 2. Line containing instrument name (must be 3 letters)
            elif instrument_name:
                current_instrument = instrument_name.group(1)
                dictionary[current_instrument] = []
            # 3. Line with list of time ranges to cut from original audio
            elif excerpt_line:
                excerpt_line = excerpt_line.group(1)
                current_excerpts = parse_excerpts_line(excerpt_line)
            # 4. Line containing a link with target file name (separated by a single space character)
            elif link_with_name:
                link, name = link_with_name.groups()
                dictionary[current_instrument].append(
                    {'title': name, 'link': link, 'excerpts': current_excerpts})
    return dictionary


def main():
    args = input_args()
    test_data = parse_file_to_dict(args.input)

    for instrument in test_data:
        # Create dir for instrument
        instrument_dir = DESTINATION_DIR / instrument
        instrument_dir.mkdir(parents=True, exist_ok=True)

        for sample in test_data[instrument]:
            sample_path = DESTINATION_DIR / \
                '{basename}.wav'.format(basename=sample['title'])
            download_sample(link=sample['link'], output_path=sample_path)

            # Divide downloaded wav file into excerpts with unique names
            for time_range in sample['excerpts']:
                start, end = time_range
                excerpt_path = instrument_dir / \
                    '{name}_{start}-{end}.wav'.format(
                        name=sample['title'], start=start, end=end)
                dump_excerpt(sample_path, excerpt_path, start, end)
                print(
                    'Successfully downloaded: {} - ({}, {})'.format(sample['title'], start, end))


if __name__ == '__main__':
    main()
