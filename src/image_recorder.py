import cv2
from multiprocessing import Barrier
from nicovision.VideoDevice import VideoDevice


class ImageRecorder:
    def __init__(self):
        self.image = None
        self.take_image = False
        self.image_barrier = Barrier(2)
        self.device = VideoDevice.from_device(
            VideoDevice.autodetect_nicoeyes()[1],
            width=1920,
            height=1440,
            zoom=150,
            tilt=-482400,
        )
        self.device.add_callback(self.create_callback())

    def __del__(self):
        if self.device.is_open():
            self.device.close()

    def close(self):
        self.device.close()

    def create_callback(self):
        def callback(rval, frame):
            """collects next frame when take_image flag for id is set"""
            if rval and self.take_image:
                self.image = frame
                self.take_image = False
                self.image_barrier.wait()

        return callback

    def record_image(self):
        self.image = None
        self.take_image = True
        self.image_barrier.wait()
        cv2.imwrite("/tmp/nico_right_eye_image.png", self.image)
        return "/tmp/nico_right_eye_image.png"
