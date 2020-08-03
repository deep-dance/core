import os
import subprocess
import math
import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description="Cut video into sequences and scale down to 800x600")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--folder', help='folder with video files')
    group.add_argument('--file', help='video file')
    parser.add_argument('--output_folder',
                        help='folder path to where the sequences will be stored',
                        default='./train')
    parser.add_argument('--cut_duration',
                            help='length in seconds of the sequences into which video(s) is(are) cut',
                            default=30, type=int)
    return parser

def get_video_duration(file):
    #get duration of video
    command_duration =['ffprobe',
              '-i', file,
              '-show_entries', 'format=duration',
              '-v', 'quiet',
              '-of', 'csv=p=0']
    duration = subprocess.check_output(command_duration)
    return duration

#scale video
def create_scaled_video(file, out_file):
    command_scale = ['ffmpeg', '-i', file, '-vf', 'scale=800:600', out_file]
    subprocess.run(command_scale)

def remove_file(file):
    command_remove = ['rm', file]
    subprocess.run(command_remove)


def cut_video_sequence(video_file, start, duration, out_folder, out_file):
    cut_command = ['ffmpeg',
                    '-ss', str(start),
                    '-i', video_file,
                    '-t', str(duration),
                    '-c', 'copy', out_folder + '/' + out_file]
    subprocess.run(['mkdir', '-p', out_folder])
    subprocess.run(cut_command)

def preprocess_video(path_to_video, output_folder, cut_duration):
    print("processing ", path_to_video)
    file = os.path.basename(path_to_video)
    print(file)
    base = os.path.splitext(file)[0]
    scaled_file = "scaled" + file
    create_scaled_video(path_to_video, scaled_file)
    duration = get_video_duration(scaled_file)
    # number of cuts
    n_seqs = math.ceil(float(duration[:-2])/cut_duration)

    # create folders and cut video into sequences
    for i in range(n_seqs):
        path_to_folder = output_folder + '/' + base + '/seq_' + str(i)
        cut_video_sequence(scaled_file, i*cut_duration, cut_duration, path_to_folder,'input_seq.mp4' )
    remove_file(scaled_file)


if __name__ == '__main__':

    args = get_parser().parse_args()

    if args.file:
        print("parsed video file")
        path_to_video = args.file
        preprocess_video(path_to_video, args.output_folder, args.cut_duration )

    elif args.folder:
        folder = args.folder
        print("parsed folder ", folder)
        for file in next(os.walk(folder))[2]:
                path_to_video = folder + '/' + file
                preprocess_video(path_to_video, args.output_folder, args.cut_duration )



    else:
        print("no path to video file nor to folder with video files specified")
