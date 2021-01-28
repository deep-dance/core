import argparse
import json
import math
import os
import subprocess



def get_parser():
    parser = argparse.ArgumentParser(
        description="Pre-process videos, targeting 800x600 @ 25fps. Cut up into smaller sequences if specified.")
    
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--input-path',
        help='Folder with video files to process.')
    
    parser.add_argument('--output-path',
        help='Folder where processed videos will be stored.')
    
    parser.add_argument('--cut',
        action="store_true",
        help='Cut video files into smaller sequences and create sub-folders to store them.')
    
    parser.add_argument('--cut-duration',
        help='Length in seconds of the sequences into which video(s) is(are) cut.',
        default=30,
        type=int)
    
    parser.add_argument('--metadata',
        action="store_true",
        help='Create JSON file containing basic metadata (input file etc.)')
    
    parser.add_argument('--debug',
        action="store_true",
        help='Print debug output.')
    
    return parser

def get_video_duration(file):
    command = ['ffprobe',
        '-i', file,
        '-show_entries', 'format=duration',
        '-v', 'quiet',
        '-of', 'csv=p=0']

    duration = subprocess.check_output(command)
    return duration

def create_scaled_video(file, out_file):
    command = ['ffmpeg',
        '-i', file,
        '-vf',
        'scale=800:600',
        out_file]

    subprocess.run(command)

def remove_file(file):
    command = ['rm', file]
    subprocess.run(command)


def cut_video_sequence(video_file, start, duration, out_folder, out_file):
    command = ['ffmpeg',
        '-ss', str(start),
        '-i', video_file,
        '-t', str(duration),
        '-c', 'copy', out_folder + '/' + out_file]

    subprocess.run(['mkdir', '-p', out_folder])
    subprocess.run(command)

def create_metadata_file(filename):
    metadata = {}
    with open(filename, 'w') as file:
        json.dump(metadata, file)

def preprocess_video(input_path, output_path, cut=False, cut_duration=30, metadata=False, debug=False):
    file = os.path.basename(input_path)
    base = os.path.splitext(file)[0]

    output_path = output_path + '/' + base
    processed_file = output_path + '/' + "input.mp4"
    metadata_file = output_path + '/' + "metadata.json"
    
    if debug:
        print("\nProcessing", file)
        print(os.path.abspath(processed_file))
        print(os.path.abspath(metadata_file))

    subprocess.run(['mkdir', '-p', os.path.abspath(output_path)])

    create_scaled_video(input_path, processed_file)

    if cut:
        duration = get_video_duration(processed_file)
        # number of cuts
        n_seqs = math.ceil(float(duration[:-2]) / cut_duration)

        # create folders and cut video into sequences
        for i in range(n_seqs):
            folder_path = output_path + '/seq_' + str(i)
            cut_video_sequence(processed_file, i * cut_duration, cut_duration, folder_path, 'input.mp4')
            create_metadata_file(metadata_file)
        remove_file(processed_file)
    else:
        if metadata:
            create_metadata_file(metadata_file)



if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.input_path:
        input_path = args.input_path
        output_path = args.output_path if args.output_path else args.input_path

        print("\nProcessing folder", os.path.abspath(input_path))

        for file in next(os.walk(input_path))[2]:
            video_path = input_path + '/' + file
            preprocess_video(video_path,
                output_path,
                args.cut,
                args.cut_duration,
                args.metadata,
                args.debug)

    else:
        print("No path to folder containing video files specified.")

    print('\n')
