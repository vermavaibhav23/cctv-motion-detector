# CCTV Motion Detection System

A computer vision mini project that highlights moving regions in a fixed webcam feed or CCTV recording. It runs locally on the CPU with Python and OpenCV; no trained model, account, or dataset is needed.

## Run the five-second demo

From this project directory, with Python 3.10 or later installed:

```sh
python -m pip install -r requirements.txt
python demo.py
python motion.py --source demo.avi --snapshots captures
```

The synthetic camera watches an empty road, a yellow vehicle crosses, then the road clears. Green rectangles mark detected motion. Expect **100 frames, one motion event, and one JPEG** in `captures/` if you let the clip finish. Press **Q** or **Esc** in the preview to stop early.

## Use your own source

| Task | Command |
| --- | --- |
| Watch the default webcam | `python motion.py` |
| Open the second camera | `python motion.py --source 1` |
| Process a CCTV recording without a window | `python motion.py --source "recording.mp4" --headless --snapshots captures` |
| Compare frame differencing on the same clip | `python motion.py --source demo.avi --method diff` |
| Ignore smaller regions | `python motion.py --source demo.avi --min-area 350` |

Snapshots are off unless you supply a directory. Each event saves its first annotated frame. Eight consecutive frames without detected motion end an event, so a continuous crossing produces one snapshot instead of dozens.

## How the detector works

1. Convert each frame to grayscale and blur it to reduce sensor noise.
2. **Background subtraction (default, MOG2):** learn a changing background, ignore the first 15 frames during calibration, and keep foreground pixels while dropping pixels classified as shadows. **Frame differencing (`diff`):** compare adjacent blurred frames and keep pixels whose brightness changes by more than 18.
3. Remove isolated pixels with morphological opening, then dilate remaining regions to connect nearby changes. Find their outer contours, discard contours below the area threshold, and draw a box around each remaining region.

The boxes describe changed regions, not object identities. Frame differencing may outline a vehicle's leading and trailing edges separately. Camera shake, sudden lighting changes, and swaying trees can trigger motion; a stationary object can blend into the learned background. Use a fixed camera and adjust `--min-area` for the resolution and distance.

## Verify the project

```sh
python -m unittest -v
```

The tests exercise both algorithms on static and moving frames, check that detections overlap the moving target, then generate and decode the demo video and verify the saved snapshots. For a classroom comparison, run the same clip with each method and explain why their boxes differ.

If a webcam cannot open, close other camera apps, confirm camera access is enabled, or try another index. On a machine without a display, use `--headless`.
