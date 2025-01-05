import threading
import time
import os
import shutil
from picamera2 import Picamera2
import cv2
os.environ["LIBCAMERA_LOG_LEVELS"] = "ERROR"  # Only log errors

class ImageCapture:
    """
    Captures images using the PiCamera2 and stores them in a specified directory.
    Runs as a separate thread.
    """

    def __init__(self):
        self.stop_event = threading.Event()
        try:
            # Try to initialize the camera
            self.picam2 = Picamera2()
        except RuntimeError as e:
            # Handle the error if the camera cannot be accessed
            print(f"Error initializing camera: {e}")
            self.picam2 = None  # Camera is not available, set to None
        self.configure_camera()

    def configure_camera(self):
        """Configures the PiCamera2 for image capture."""
        if self.picam2:
            config = self.picam2.create_still_configuration(main={"size": (640, 480)})
            # camera_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(2)  # Allow the camera to initialize
            print("Camera configured successfully.")
        else:
            print("Camera not available, cannot configure.")

    def get_frame(self):
        """Returns the latest frame captured by the camera."""
        if not self.picam2:
            print("Camera not available, cannot get frame.")
            return None
        else:
            return self.picam2.capture_array()
        
    def stop(self):
        """Stops image capturing and releases the camera."""
        self.stop_event.set()
        self.picam2.stop_preview()
        self.picam2.close()

if __name__ == "__main__":
    # create an instance of the ImageCapture class
    cam = ImageCapture()
    print("ImageCapture instance created.")

    # capture image and save it with numpy 
    im = cam.get_frame()
    print(f"Captured image: {im.shape}")

    # save as png with opencv
    cv2.imwrite("image.png", im)
    print("Image saved as image.png")