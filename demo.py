"""Generate a five-second synthetic CCTV clip. No downloads required."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def generate(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 360))
    if not writer.isOpened():
        raise OSError(f"Cannot create {path}; use an AVI path and check codec support")
    try:
        for i in range(100):
            frame = np.full((360, 640, 3), (51, 47, 42), np.uint8)
            cv2.rectangle(frame, (0, 65), (639, 280), (77, 76, 74), -1)
            for x in range(0, 640, 90):
                cv2.line(frame, (x, 174), (x+42, 174), (180, 182, 180), 3)
            cv2.putText(frame, "CAM 04   /   EAST GATE", (20, 326),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (215, 218, 213), 1, cv2.LINE_AA)
            if 24 <= i < 84:
                x = (i-24)*12 - 56
                cv2.rectangle(frame, (x, 192), (x+55, 230), (40, 178, 246), -1)
                cv2.rectangle(frame, (x+33, 196), (x+44, 225), (54, 65, 71), -1)
            writer.write(frame)
    finally:
        writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="demo.avi")
    args = parser.parse_args()
    generate(args.path)
    print(f"Wrote {args.path} (100 frames at 20 fps)")
