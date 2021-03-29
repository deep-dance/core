# deep.dance | core

## Get started

The project is seperated into a few different parts:

- Detector: Predict 3d poses in single camera videos for custom dataset creation
- Creator: RNN sequence generator
- Renderer: Blender project which animates a character with generated sequences

### Mining Movement: The Detector

The project comes with a few sample sequences which already went through the process of 3D pose prediction.
The following section will explain how the pool of sample sequences can be extended with custom video sequences.

#### Prepare environment

##### 1. Setup DVC

We're using [DVC](https://dvc.org/). Please install the latest version.

###### Directories

The following directories are managed by DVC (look out for `.dvc` file):

```
data/train/video/example
data/train/video/deep-dance
```

###### Remotes

Currently, there is only one remote configured, which works on a server with a locally shared folder between users:

```
/home/shared/dvc
```

If that folder does not exist, DVC can't pull the files.

###### Workflow

The DVC workflow is very similar to Git. In a project folder, you could initialize and use DVC with

```
$ dvc init
$ dvc add data/*
$ dvc remote add -d local /home/shared/dvc
$ dvc run -d data -o model.p train.py
$ dvc push
```

##### 2. Clone repository

```
$ git clone --recursive https://github.com/deep-dance/core.git
```

The repository reflects the parts mentioned above and seperates them into data, scripts, or other executables.

After cloning the respository, all data, managed by DVC also needs to be pulled. 

```
$ dvc pull
```

##### 3. Setup Docker container

Note, that the Docker container only works for a setup with nvidia graphic cards that can run cuda.
See [Docker instructions](DOCKER.md).

##### 5. Setup VideoPose3D

Make sure the folder `VideoPose3D` was cloned properly and download the pretrained model for 3D pose estimation.

```
cd VideoPose3D
mkdir checkpoint && cd checkpoint
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin
```
for trajectory tracing also download addidional models [here](https://drive.google.com/file/d/1kJKDjdpFcg7cXr3x_hV3lYL0Tm3ImsFY/view?usp=sharing)
This link was posted by one of the maintainers of VideoPose3D in the issue comments [here](https://github.com/facebookresearch/VideoPose3D/issues/145)

#### Run on custom videos

This assumes the data structure to be 
```
data/train/video/deep-dance/<dancer>/<tag-list>/input.mp4
data/train/video/deep-dance/<dancer>/<tag-list>/metadata.json
data/train/video/deep-dance/<dancer>/<tag-list>/pose2d.npz
data/train/video/deep-dance/<dancer>/<tag-list>/pose3d.mp4
```
It is recommended to prepare videos to match the Human3,6m data set as closely as possible. Video sequences should be saved in the folder mentioned above and should have a length of 30 seconds.

##### Preprocess videos

###### Motion database

It's required to setup a motion database that reflects the dataset and contains metadata on every video in it. The current workflow requires the motion database to include the following metadata: name of the dancer and mapping of a video name to a list of tags. Both will be used while creating the dataset from custom custom videos. The motion database needs to be stored in a JSON file:

```
{
    "dancers": [
        {
            "name": "adancer",
            "videos": {
                "V0001": [ "normal", "spot", "frontal", "standing" ]
            }
        }
    ]
```

There's a script that is used to preproces and store raw video files into the format explained above and it can be called like this:

```
python3 data/preprocess_videos.py --input-path videos/adancer --output-path train/video/deep-dance/adancer --database motion-db.json --metadata
```

The script takes an input path from which all video file are being picked up, scaled down and saved together with their metadata file. The folder structure that's being created depends on the tags given by the motion database. The metadata file contains a path to the original video it was created from and the tags given by the motion database.


###### ffmpeg helpers

Cut videos into sequences of 30 seconds, e.g.

```
ffmpeg -i input_raw.mp4 -ss 3:55 -to 4:25 -c copy data/train/seq_[index]/input_seq.mp4
```

It is desired to crop input videos such that they match Human3.6m in terms of the ratio between movement area and background. It still needs to be researched, if this is really necessary, or if the pose estimation results are still good enough without this step. Reducing the size of the input video has a positive effect on computation time in any case.

```
ffmpeg -i input_seq.mp4 -filter:v "crop=out_w:out_h:x:y" data/train/seq_[index]/input.mp4
```

    out_w is the width of the output rectangle
    out_h is the height of the output rectangle
    x and y specify the top left corner of the output rectangle

Resize videos:

```
ffmpeg -i input_seq.mp4 -vf scale=800:600 output.avi
```

##### Run 2D pose estimation

Before running 3D pose estimation, 2D poses need to be estimated. The following command will run the detectron2 predictor on all video files named `input.mp4` in `data/train/seq_[index]` and create the numpy archive `pose2d.npz`, which contains detected boxes and keypoints. It does not overwrite exising `pose2D.npz` files, unless its told so by passing `--overwrite`.

---
**Note:**
By default the `COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x` model is used inside the detectron2 layer. Please refer to their documentation for other pretrained models and change paths in `detector/predict_2d_pose.py` accordingly.

---

```
python predict_2d_pose.py
```

##### Run 3D pose estimation

```
python predict_3d_pose.py
```

### Work for me: The Creator

#### Train model

The workflow is using [DVC pipelines](https://dvc.org/doc/start/data-pipelines). All stages are included in `dvc.yaml`. **Note:** everything needs to happen in the Tensorflow Docker container.

The stage **prepare** was added issuing this:

```
dvc run -n prepare \
    -p prepare.split,prepare.look_back \
    -d src/prepare.py \
    -o ../data/train/prepared \
    python src/prepare.py
```

The stage **train** was added issuing this:

```
dvc run -n train \
    -p train.data,train.epochs,train.batch_size,train.look_back,train.lstm_layer,train.mdn_layer,train.custom_loss \
    -d src/train.py \
    -M ../data/metrics/deep-dance-scores.json \
    --plots-no-cache ../data/metrics/deep-dance-loss.json \
    -o ../data/models \
    -o ../data/metrics \
    python src/train.py
```

The stage **generate** was added issuing this:

```
dvc run -n generate \
    -p generate.seed,generate.steps_limit,generate.random_state,generate.look_back,generate.custom_loss \
    -d src/generate.py \
    -o ../data/generated \
    python src/generate.py
```

There's no need to use dvc add for DVC to track stage outputs (`../data/train/prepared/models` in this case); `dvc run` already took care of this. You only need to run `dvc push` if you want to save them to remote storage, (usually along with `git commit` to version `dvc.yaml` itself).

##### Experiments

`dvc run exp` needs a working Git config (because it could also push results to a Git remote directly). Therefor it is necessary to set a username and an email:

```
git config --global user.name "username"
git config --global user.email "username@mail.me"
```

###### Troubleshooting

1. `cannot stash changes - there is nothing to stash.`

One small workaround I have for this at the moment is make a small change to any file that is under git version control (but not related to any of the stages,etc). See issue (here)[https://github.com/iterative/dvc/issues/5684].

##### Debugging

#### Simple matplot rendering
```
python3.6 render_pose3d_matplot.py ../data/train/seq_001/pose3d.npz --frames 90
```

### Rotating Bones: The Renderer

Several experiments with Blender have been done without any ground-breaking results. A component which translates keypoints into skeletal bone rotations for Blender needs to developed.

## Contact

### Maintainer

@erak
@nikozoe
