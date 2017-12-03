# Behaviorial Cloning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="writeup-images/overview.PNG" width="1000" alt="Combined Image" />

Overview
---
In this project, we will use deep neural networks and convolutional neural networks to clone driving behavior. We will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

Goals
---
- Step01: The model can be used to successfully operate the simulation.
- Step02: Using Python generator in in model.py to generate data with code organized and commented.
- Step03: Using Convolution layers, filter, nonlinearity, data normalizarion into the model.
- Step04: Using Train/validation/test for splits and methods to reduce overfitting in the model(dropout/others).
- Step05: Using learning rate or Adam optimizer for overfitting.
- Step06: Training data chosen to keep the car on the track.
- Step07: Document the approach for deriving and designing a solution(model architecture fit).
- Step08: Document the model architecture type used, layers, size using visualizations each qualities.
- Step09: Document and provide sample how the model was trained and its dataset characteristics.
- Step10: In Simulator No tire may leave the drivable portion of the track surface.
- Step11: Can the model work for car to stay on the road for *track two* as well

Approach
---
- Data: Use the simulator to collect data of good driving behavior.
- Model: Design, train and validate a model to predicts a steering angle from image data.
- Test: Use the model to drive the vehicle autonomously around the first track in the simulator. 
- Document: Summarize the results in an written report.
</br> **NOTE: The vehicle should remain on the road for an entire loop around the track.**

Result
---
 Track1                     |  Track2
 :-------------------------:|:-------------------------:
 <img src="writeup-images/track1.gif" width="320" height="160" alt="Track1" /> |  <img src="writeup-images/track2.gif" width="320" height="160" alt="Track1" />

Required Files
---
- [model.py](./model.py) (script used to create and train the model)
- [drive.py](./drive.py) (script to drive the car - this file is unchanged)
- [model.h5](./model.h5) (a trained Keras model)
- [README.md](./README.md) (a report writeup markdown file)
- [video.mp4](./video.mp4) (a video recording of vehicle driving autonomously around the track for at least one full lap)

NOTE:
* simulator

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

