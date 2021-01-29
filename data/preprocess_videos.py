import argparse
import json
import math
import os
import subprocess



def get_parser():
    parser = argparse.ArgumentParser(
        description='Pre-process videos, targeting 800x600 @ 25fps. ' +
            'Reads additional information from JSON database.')
    
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument('--input-path',
        help='Folder with video files to process.')
    
    parser.add_argument('--output-path',
        help='Folder where processed videos will be stored.')
    
    parser.add_argument('--database',
        help='JSON file that stores the motion database.',
        required=True)
    
    parser.add_argument('--metadata',
        action='store_true',
        help='Create JSON file containing basic metadata (input file etc.)')
    
    parser.add_argument('--debug',
        action='store_true',
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

def create_metadata_file(filename, video_path, dancer_name, base, database):
    metadata = {}
    metadata['file'] = []
    metadata['file'].append({
        'source': video_path,
    })

    dataset = find_dataset(dancer_name, database)
    if dataset:
        try:
            metadata['info'] = []
            metadata['info'].append({
                'tags': dataset['videos'][base],
            })
        except KeyError:
            print('Key not found')
    with open(filename, 'w') as file:
        json.dump(metadata, file)

def read_database_file(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        return data

def build_folder_name(dancer_name, filename, database):
    dataset = find_dataset(dancer_name, database)
    if dataset:
        try:
            tags = dataset['videos'][filename]
            folder_name = '_'.join(tags)
            return folder_name    
        except KeyError:
            print('Key not found')

    return ''

def find_dataset(dancer_name, database):
    for dataset in database['dancers']:
        if dataset['name'] == dancer_name:
            return dataset

    return false


def preprocess_video(video_path, input_path, output_path, database,
    metadata=False, debug=False):
    
    file = os.path.basename(video_path)
    base = os.path.splitext(file)[0]
    dancer_name = os.path.basename(os.path.normpath(input_path))
    folder_name = build_folder_name(dancer_name, base, database)

    if folder_name:
        output_path = output_path + '/' + folder_name
        processed_file = output_path + '/' + 'input.mp4'
        metadata_file = output_path + '/' + 'metadata.json'
        
        if debug:
            print('Processing', file)
            print(os.path.abspath(processed_file))
            print(os.path.abspath(metadata_file))

        subprocess.run(['mkdir', '-p', os.path.abspath(output_path)])

        create_scaled_video(video_path, processed_file)

        if metadata:
                create_metadata_file(metadata_file,
                    os.path.abspath(video_path),
                    dancer_name,
                    base,
                    database)
    else:
        print('No entry in database found for ', dancer_name, '. Aborting.')
        
if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.input_path:
        input_path = args.input_path
        output_path = args.output_path if args.output_path else args.input_path

        print('\n')
        print('Reading database file', args.database)
        print('Processing folder', os.path.abspath(input_path))

        database = read_database_file(args.database)

        for file in next(os.walk(input_path))[2]:
            video_path = input_path + '/' + file
            preprocess_video(video_path,
                input_path,
                output_path,
                database,
                args.metadata,
                args.debug)

    else:
        print('No path to folder containing video files specified.')

    print('\n')
