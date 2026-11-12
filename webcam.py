"""Live webcam preview for OpenCV installations without window support."""
import argparse
import tkinter as tk

import cv2

from motion import Detector


def preview(index):
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(index)
    try:
        if not camera.isOpened():
            raise SystemExit("Cannot open camera. Close other camera apps or try --camera 1.")
        window = tk.Tk()
        window.title("Live motion detection | Q or Esc to close")
        view = tk.Label(window)
        view.pack()
        detector = Detector()

        def update():
            ok, frame = camera.read()
            if not ok:
                print("Camera stopped sending frames.", flush=True)
                window.destroy()
                return
            boxes = detector.detect(frame)
            for x, y, w, h in boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (70, 235, 110), 2)
            status = "MOTION" if boxes else "CLEAR"
            if detector.frames <= 15:
                status = "CALIBRATING"
            cv2.putText(frame, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (70, 235, 110), 2, cv2.LINE_AA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            header = f"P6\n{width} {height}\n255\n".encode()
            photo = tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")
            view.configure(image=photo)
            view.image = photo
            if detector.frames == 1:
                print("Live preview started. Press Q or Esc to close.", flush=True)
            window.after(15, update)

        window.bind("<Escape>", lambda event: window.destroy())
        window.bind("<KeyPress-q>", lambda event: window.destroy())
        window.after(0, update)
        window.mainloop()
    finally:
        camera.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    preview(parser.parse_args().camera)
