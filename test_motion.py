import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from demo import generate
from motion import Detector, run


class MotionTests(unittest.TestCase):
    def test_quiet_and_localized_motion(self):
        for method in ("diff", "mog2"):
            with self.subTest(method=method):
                detector = Detector(method)
                frame = np.zeros((120, 160, 3), np.uint8)
                for _ in range(20):
                    self.assertEqual(detector.detect(frame), [])
                frame[45:75, 60:95] = 255
                boxes = detector.detect(frame)
                self.assertTrue(any(x <= 75 < x+w and y <= 60 < y+h for x,y,w,h in boxes))

    def test_video_and_snapshots(self):
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "demo.avi"
            generate(clip)
            for method in ("mog2", "diff"):
                folder = Path(tmp) / method
                with self.subTest(method=method):
                    self.assertEqual(run(str(clip), method, 180, folder, True), (100, 1))
                    images = list(folder.glob("*.jpg"))
                    self.assertEqual(len(images), 1)
                    self.assertEqual(cv2.imread(str(images[0])).shape, (360, 640, 3))


if __name__ == "__main__":
    unittest.main()
