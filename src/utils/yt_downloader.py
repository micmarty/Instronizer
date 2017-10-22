import os
import sys
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
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')
    if d['status'] == 'downloading':
        sys.stdout.write("\033[F")  # Cursor up one line
        print(d['filename'][:70] + '...', d['_percent_str'])


def exit_gracefully_handler(signum, frame):
    print('Caught SIGINT: Exiting gracefully...')
    exit()


class YT_downloader():

    if __name__ == '__main__':
        MODULE_DIRECTORY = os.getcwd()  # remember directory we're in to restore it later
        signal.signal(signal.SIGINT, exit_gracefully_handler)
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

        args = parser.parse_args()
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if args.output_dir[-1:] == '/':  # remove trailing slash
            args.output_dir = args.output_dir[:-1]
        current_directory = args.output_dir
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'logger': YdlLogger(),
            'progress_hooks': [ydl_hook],
        }

        with open(args.input_file) as f:
            for line in f:
                # comment line
                if line[:1] == '#':
                    continue
                # directory line
                elif line[:1] == '[':
                    relative_directory = line[1:-2]  # trim braces
                    # add a slash if it's not the first character in path
                    if not relative_directory[1:2] == '/':
                        relative_directory = '/' + relative_directory
                    current_directory = args.output_dir + relative_directory
                    if not os.path.exists(current_directory):
                        os.makedirs(current_directory)
                    try:
                        os.chdir(current_directory)
                    except OSError:
                        print('Could not enter directory ' + current_directory + ' ,ensure that all paths \
								in the input file are correct')
                        exit()
                # link line
                else:
                    ext = '.wav'
                    split_line = line.split()
                    link = split_line[0]
                    if len(split_line) < 2: #desired file name was not specified (the list has only one element)
                        song_filename = 'video without a specified title'
                        ydl_opts.pop('outtmpl') #removing this value from ydl_opts makes youtube-dl download into current
                                                #working directory, with video title as filename
                    else:
                        song_filename = split_line[1]
                        ydl_opts.update({'outtmpl': current_directory + '/' + song_filename + ext})
                    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                        video_info = ydl.extract_info(link, download=False)
                        video_title = video_info.get('title', None)
                        video_id = video_info.get('id', None)
                        if not os.path.isfile(current_directory + '/' + video_title + '-' + video_id + ext) \
                            and not os.path.isfile(current_directory + '/' + song_filename + ext):
                            print('Downloading ' + song_filename)
                            ydl.download([link])
                        else: 
                            print('Audio from ' + video_title[:30] + '[...] already exists in directory:\n' +
                                current_directory + ' ,skipping')

        os.chdir(MODULE_DIRECTORY)
