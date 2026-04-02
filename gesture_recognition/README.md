# Webcam Hand Gesture Recognition (MediaPipe)

This module runs a **MediaPipe Gesture Recognizer** model using a webcam
or USB camera and prints the estimated hand gesture to standard output.

It is designed to work with **custom `.task` gesture models** exported
from MediaPipe Model Maker or other MediaPipe pipelines.

The script captures frames from a camera, runs gesture recognition, and
outputs the **top predicted gesture label and score**.

Example output:

OPEN_PALM 0.932 NONE FIST 0.874

This can be easily integrated into other systems such as robotics
pipelines, control interfaces, or logging tools.

------------------------------------------------------------------------

# Requirements

Python 3.12 is recommended.

Install dependencies using:

pip install -r requirements.txt

Example `requirements.txt`:

mediapipe\>=0.10.13 opencv-python\>=4.9.0 numpy\>=1.26

------------------------------------------------------------------------

# Script Overview

The script:

1.  Opens a webcam or USB camera using OpenCV
2.  Loads a MediaPipe Gesture Recognizer `.task` model
3.  Processes each frame
4.  Prints the top gesture classification to stdout

No GUI is required.

------------------------------------------------------------------------

# Usage

Basic usage:

python gesture_webcam.py --model path/to/model.task

This will start the camera and continuously print gesture predictions.

Stop the program with:

Ctrl + C

------------------------------------------------------------------------

# Command Line Arguments

## --model

Path to the MediaPipe gesture recognition `.task` model.

Required.

Example:

--model models/gesture_recognizer.task

------------------------------------------------------------------------

## --camera

Camera index used by OpenCV.

Default: 0

Most laptops use index `0` for the built-in webcam. If you have multiple
cameras, try `1`, `2`, or higher.

Example:

python gesture_webcam.py --model gesture.task --camera 1

------------------------------------------------------------------------

## --mirror

Mirrors the camera feed horizontally before processing.

Useful for webcam-style interaction where movements should feel natural.

Example:

python gesture_webcam.py --model gesture.task --mirror

------------------------------------------------------------------------

## --print-on-change

Only prints output when the predicted gesture changes.

This reduces console spam during continuous detection.

Example:

python gesture_webcam.py --model gesture.task --print-on-change

------------------------------------------------------------------------

## --min-score

Minimum confidence score required to print a gesture prediction.

Predictions below this value will be ignored.

Example:

python gesture_webcam.py --model gesture.task --min-score 0.6

------------------------------------------------------------------------

# Example Commands

Basic detection:

python gesture_webcam.py --model gesture.task

Use a USB camera:

python gesture_webcam.py --model gesture.task --camera 1

Mirror the webcam feed:

python gesture_webcam.py --model gesture.task --mirror

Reduce console spam:

python gesture_webcam.py --model gesture.task --print-on-change

Filter low confidence predictions:

python gesture_webcam.py --model gesture.task --min-score 0.7

Full example:

python gesture_webcam.py\
--model gesture.task\
--camera 0\
--mirror\
--print-on-change\
--min-score 0.6

------------------------------------------------------------------------

# Output Format

Each detection prints a single line:

GESTURE_LABEL CONFIDENCE

Example:

OPEN_PALM 0.923

If no gesture is detected:

NONE

This format makes it easy to pipe into other programs.

Example:

python gesture_webcam.py --model gesture.task \| tee gestures.log

------------------------------------------------------------------------

# Project Structure

project/ │ ├── gesture_webcam.py ├── requirements.txt ├── README.md └──
models/ └── gesture.task

------------------------------------------------------------------------

# Notes

-   The script uses **MediaPipe Vision Tasks**
-   Any `.task` gesture model compatible with MediaPipe Gesture
    Recognizer can be used
-   Performance depends on your camera resolution and system CPU or GPU

------------------------------------------------------------------------

# Possible Extensions

Some common improvements include:

-   Publishing gesture results to ROS topics
-   Sending gestures over WebSocket
-   Triggering robot actions based on detected gestures
-   Recording gesture sequences for training datasets

------------------------------------------------------------------------

# License

This module can be distributed under any license appropriate for your
project. MediaPipe itself is released under the Apache 2.0 license.
