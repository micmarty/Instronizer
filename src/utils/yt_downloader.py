from pathlib import Path
import signal
import argparse
import textwrap
import youtube_dl

class YdlLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def ydl_hook(d):
    '''Actions triggered by yotube_dl on specific events'''
    if d['status'] is 'finished':
        print('Done downloading, now converting...\n')
    elif d['status'] is 'downloading':
        # Print progress in-place
        print('{filename}... => {progress}'.format(
            filename=d['filename'][:70], progress=d['_percent_str']), end='\r')


def exit_gracefully_handler(signum, frame):
    '''<Ctrl + c> handler'''
    print('Caught SIGINT: Exiting gracefully...')
    exit()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Downloader and extractor of Wave audio from Youtube videos.',
        usage='python yt_downloader.py -i <formatted_input_file> -o <output_directory>',
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-i', '--input-file',
                        action='store',
                        required=True,
                        help=textwrap.dedent('''\
                                An text file containing links to be downloaded. The file MUST be formatted 
                                in the following way (.wav will be appended to the desired filename):\n
                                [path_relative_to_output_dir_argument1] 
                                https://YT_URL1 desired_filename1 
                                https://YT_URL2 desired_filename2 
                                https://YT_URL3 desired_filename3\n
                                [path_relative_to_output_dir_argument2]
                                https://YT_URL4 desired_filename4
                                https://YT_URL5 desired_filename5
                                https://YT_URL6 desired_filename6
                                .
                                .
                                .
                                etc. Desired filename may be empty, in which case youtube video title is used
                                as filename\n
                                '''))

    parser.add_argument('-o', '--output-dir',
                        action='store',
                        required=True,
                        help='A directory where downloaded audio will be saved in')

    return parser.parse_args()

class YT_downloader():
    def __init__(self, args):
        self.curr_workdir = Path.cwd()

        self.input_file_path = args.input_file
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_path = None

        self.target_format = 'wav'
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.target_format,
            }],
            'logger': YdlLogger(),
            'progress_hooks': [ydl_hook],
        }

    def parse_file_and_download(self):
        '''Parse 3 types of lines in given formatted file'''
        with open(self.input_file_path) as f:
            for line in f:
                # 1. Ignore comment or empty line
                if line.startswith('#') or not line.strip():
                    continue
                # 2. Target directory line
                elif line.startswith('['):
                    # Get the text between '[' and ']' and create missing directories
                    self.target_path = self.output_dir / Path(line[1:-2].strip())
                    self.target_path.mkdir(parents=True, exist_ok=True)
                # 3. Line containing link or link with target file name
                elif line.startswith('http'):
                    line_parts = line.split()
                    self._parse_link_line(line_parts)
                    

    def _parse_link_line(self, line_parts):
        song_filename = None

        # Desired file name was not specified (the list has link only)
        if len(line_parts) < 2:
            link = line_parts[0]
            with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
                song_filename = ydl.extract_info(
                    link, download=False).get('title')
        # Both, link and file name were specified
        elif len(line_parts) is 2:
            link, song_filename = line_parts
        
        target_path = (self.target_path / song_filename).with_suffix('.' + self.target_format)
        self.ydl_opts.update({'outtmpl': str(target_path)})
        self._download(link, target_path)
        
    def _download(self, link, target_path):
        with youtube_dl.YoutubeDL(self.ydl_opts) as ydl:
            print('Downloading {}'.format(target_path.name))
            ydl.download([link])


if __name__ == '__main__':
    # <Ctrl + c> event handling
    signal.signal(signal.SIGINT, exit_gracefully_handler)
    args = parse_args()
    yt_downloader = YT_downloader(args)
    yt_downloader.parse_file_and_download()
