"""Motion detection for a fixed webcam or CCTV recording."""
import argparse
from datetime import datetime
from pathlib import Path

import cv2


class Detector:
    def __init__(self, method="mog2", min_area=180):
        self.previous = None
        self.background = (cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=24, detectShadows=True
        ) if method == "mog2" else None)
        self.min_area = min_area
        self.frames = 0

    def detect(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self.frames += 1
        if self.background is not None:
            mask = self.background.apply(gray)
            if self.frames <= 15:  # Let the model see the scene before alerting.
                return []
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        else:
            previous, self.previous = self.previous, gray
            if previous is None:
                return []
            _, mask = cv2.threshold(cv2.absdiff(previous, gray), 18, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= self.min_area]


def run(source, method, min_area, snapshots, headless):
    detector = Detector(method, min_area)
    cap = cv2.VideoCapture(int(source) if source.isdecimal() else source)
    session = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    active, events, quiet = False, 0, 0
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open source. Check the video path or camera index.")
        while True:
            ok, frame = cap.read()
            if not ok:
                if not detector.frames or source.isdecimal():
                    raise ValueError("The source stopped returning readable frames.")
                break
            boxes = detector.detect(frame)
            for x, y, w, h in boxes:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (70, 235, 110), 2)
            label = "MOTION" if boxes else "CLEAR"
            if method == "mog2" and detector.frames <= 15:
                label = "CALIBRATING"
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (70, 235, 110), 2, cv2.LINE_AA)
            if boxes:
                if not active:
                    events += 1
                    print(f"Event {events}: motion at frame {detector.frames}")
                    if snapshots:
                        snapshots.mkdir(parents=True, exist_ok=True)
                        path = snapshots / f"{session}-event-{events:03}.jpg"
                        if not cv2.imwrite(str(path), frame):
                            raise OSError(f"Cannot write snapshot: {path}")
                active, quiet = True, 0
            else:
                quiet += 1
                if quiet >= 8:
                    active = False
            if not headless:
                cv2.imshow("CCTV motion detector | Q to quit", frame)
                fps = cap.get(cv2.CAP_PROP_FPS)
                delay = max(1, round(1000/fps)) if not source.isdecimal() and 0 < fps < 1000 else 1
                if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):
                    break
    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()
    print(f"Processed {detector.frames} frames; motion events: {events}")
    return detector.frames, events


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="0", help="Webcam index or video file path")
    p.add_argument("--method", choices=["mog2", "diff"], default="mog2")
    p.add_argument("--min-area", type=int, default=180, help="Minimum contour area in pixels")
    p.add_argument("--snapshots", type=Path, help="Save the first annotated frame of each event here")
    p.add_argument("--headless", action="store_true", help="Process without a preview window")
    args = p.parse_args()
    if args.min_area < 1:
        p.error("--min-area must be positive")
    try:
        run(**vars(args))
    except (OSError, ValueError, cv2.error) as exc:
        p.exit(1, f"Error: {exc}\n")
