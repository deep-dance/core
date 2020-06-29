# TO SHARE - Technical Concept

## Motivation

Develop a system that can generate a dance choreography of variable length for multiple performers. The output should be a 3D rendering which can be used by every individual dancer to study their part, e.g.:


| Frame 1 | Frame n |
| ------- | ------- |
| [![](https://img.youtube.com/vi/c9h9zc7uPWQ/0.jpg)](https://www.youtube.com/watch?v=c9h9zc7uPWQ) | [![](https://img.youtube.com/vi/Q4_XSMqN8w0/0.jpg)](https://www.youtube.com/watch?v=Q4_XSMqN8w0) |


## Overview

Research on generative algorithms suited for this project revealed that a framework containing (a) neural network(s) could be used to achieve the desired outcome. Such a framework could be trained with existing choreographic sequences (video or motion capture) to generate new sequences. As already mentioned, neural networks usually require a training phase before they can generate new output or classify things etc. Therefor the system described here, reflects those phases.

![](https://hack.borg.sh/uploads/upload_f88e14040960cedadf95fb74524a7696.png)

### Keypoints

It is desired to design a system that works on so called keypoints, as the basic data structure for all phases. A keypoint reflects a joint in the human body, and sequences of keypoints can be easily retreived from pose recognition software ran on images or by using a Kinect to do a basic motion capture.
Choreographic sequences used to build this system and also as output of such, are described with keypoints. A sequence is just a list of keypoints.

<!-- ![](https://hack.borg.sh/uploads/upload_891f37273441085b50783968c5018490.png) -->


### Other notations

Alternatively to keypoints, an existing dance notation, e.g. Laban notation, could have been used as the base for all phases. But since this would require additional image recogition and 3D rendering code specifically written for such a visual notation, keypoints were chosen to be the base data structure for this project.


![](https://hack.borg.sh/uploads/upload_50b40b4d8caa836d67a246929a970bd4.jpeg)


## Phases

The above led to a system design that is split into multiple phases. Each phase is then implemented using one or more of the compoments described later in this section.

### Phase 1: Obtain training data

All motion generation is depending on a good training set. Good in this case means sufficiently large and also reflecting the style of the motions wanted.

This can only be achieved be recording dancers performing certain qualities and simple as well as complex movements [[1](#1)].

#### Image / video-based

Image or video-based keypoint extraction 

##### OpenPose [[2](#2)]

Since it is required to record the same scene from multiple views in order to perform 3D reconstruction, a test arrangement, ideally built in a rehearsal environment, needs to be created. This should include 4 Cameras, arranged on the rehearsal stage.

##### posenet-tf

https://www.tensorflow.org/lite/models/pose_estimation/overview
https://github.com/deephdc/posenet-tf

##### Radical

https://getrad.co/studio-product/

##### posenet-python

https://www.tensorflow.org/lite/models/pose_estimation/overview
https://github.com/atomicbits/posenet-python

#### Kinect motion capture

https://www.youtube.com/watch?v=GPjS0SBtHwY


### Phase 2: Train system

A lot of groundwork has been done by Luka Crnkovic-Friis and Louise Crnkovic-Friis outlined in [Generative Choreography using Deep Learning](https://arxiv.org/pdf/1605.06921.pdf).

### Phase 3: Generate and render motion sequences

Store motion sequences in plain text files and create tool that can render them.

### Components

This section desribes a first draft of a system that could perform all phases. Components share interfaces which make them interchangeable.

#### Phase 1

##### Phase 1.1: Obtain training data with OpenPose

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is an open-source toolkit which allows human pose estimation. It uses a pre-trained neural network to find keypoints in images and videos. If used with multiple cameras, captering multiple views of the same scene, it is able to perform 3D reconstruction. The output of OpenPose are JSON files which contain detected keypoints:

| Still     | OpenPose result |
| --------  | --------------- |
| ![](https://hack.borg.sh/uploads/upload_53e98633c6d83fe9b960989e03f7d3fe.png) | ![](https://hack.borg.sh/uploads/upload_3bb0bfaf9fb4d91ee7cb7875594c54d9.png) |
|![](https://hack.borg.sh/uploads/upload_005d99c52d869c4b83f2130950bf7db3.png) | ![](https://hack.borg.sh/uploads/upload_886428a23863149e0147c7d3727add22.png) |

##### Phase 1.1.1: Record dance in test arrangement

Since OpenPose requires the same scene recorded from multiple views in order to perform 3D reconstruction, a test arrangement, ideally built in a rehearsal environment, needs to be created. This should include 4 Cameras, arranged on the rehearsal stage.

All videos will be tagged and archived for further analysis.

##### Phase 1.1.2: Feed videos into OpenPose and run 3D reconstruction

##### Phase 1.2: Output transform

The output of OpenPose, which is JSON, needs to be transformed in order to be read in phase 2. The JSON output needs to be parsed and only keypoints detected by the 3d reconstruction should be saved in the final output.

The transformation should create a format that is easily translatable to tensors.

#### Phase 2: Own implementation of chor-rnn paper

Try implementing the method outlined in [Generative Choreography using Deep Learning](https://arxiv.org/pdf/1605.06921.pdf) by Peltarion (Luka Crnkovic-Friis and Louise Crnkovic-Friis).

The re-implementation of chor-rnn (our name here) should then produce motion sequences and output them in a simple CSV format.

#### Phase 3: Custom 3D rendering

#### tool based on Kiss3D

Even though rendering of sequences is a relatively simple task, it needs to be implemented early and should come with as less overhead as possible, such that the focus can stay on the actual training and learning process.

Therefor a simple rendering stage witten in Rust and based on the [Kiss3D](http://kiss3d.org/) engine was choosen.

##### CSV input

The naive in memory size estimation $Size_{Full}$ in KB of the CSV input file for one skeleton on 64-bit System

$Size_{Row} = Num_{Keypoints} * 3 * 64bit$
$Size_{Full} = Size_{Row} * Framerate * 60 * duration$

In case of a 1 hour sequence at 25 fps:

$Framerate = 25$
$Num_{Keypoints} = 25$

$Size_{Row} = Num_{Keypoints} * 3 * 64bit$
$Size_{Row} = 0,59 kByte$

$Size_{Full} = Size_{Row} * Framerate * 60 * duration$
$Size_{Full} = 51,9MByte$

## References

###### [1]
https://nanonets.com/blog/human-pose-estimation-2d-guide/

###### [2]
https://github.com/CMU-Perceptual-Computing-Lab/openpose

###### [3]

###### [4]


###### [5]





