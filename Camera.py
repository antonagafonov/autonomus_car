import time
import os
import cv2  # For web camera support
import numpy as np
import multiprocessing as mp
from picamera2 import Picamera2, Metadata
from utils import get_time

os.environ["LIBCAMERA_LOG_LEVELS"] = "ERROR"  # Only log errors


class ImageCapture:
    """
    Captures images using the PiCamera2 or a web camera in a separate process.
    The latest image is always stored in shared memory.
    """

    def __init__(self, size=(640, 480),offset = 30):

        # Shared memory and synchronization primitives
        self.manager = mp.Manager()
        self.shared_frame = self.manager.list([None])  # Initialize with None
        self.shared_timestamp = self.manager.Value("d", 0.0)  # Shared memory for timestamp
        self.lock = mp.Lock()
        self.stop_event = self.manager.Event() 
        # Process for capturing images
        self.process = mp.Process(target=self._capture_process)
        self.size = size
        self.offset = offset
        
    def configure_camera(self):
            """Configures the PiCamera2 for image capture."""
            if self.camera:
                config = self.camera.create_video_configuration(
                main={"size": (640, 480)},
                # main={"size": (640, 480), "format": "YUV420"},
                controls={
                    "FrameRate": 30,            # Target 90 FPS
                    "ExposureTime": 30000,       # Reduce exposure to allow higher FPS
                    "AnalogueGain": 1.0,        # Fix gain to prevent auto adjustments
                    "AwbEnable": True,         # Disable Auto White Balance
                    "AeEnable": True,          # Disable Auto Exposure
                    "FrameDurationLimits": (5000, 20000),  # Min exposure to allow 90 FPS
                })
                self.camera.configure(config)
                self.camera.start()
                print("Camera configured successfully.")
                metadata = Metadata(self.camera.capture_metadata())
                print(metadata.ExposureTime, metadata.AnalogueGain)
            else:
                print("Camera not available, cannot configure.")

    def list_available_sizes(self):
        """Lists all available resolutions for the PiCamera2."""
        if self.camera:
            camera_info = self.camera.camera_properties
            resolutions = camera_info.get("PixelArrayActiveAreas", [])
            print("Available resolutions:", resolutions)
        else:
            print("Camera not available.")

    def start_capturing(self):
        self.process.start()
        
    def _capture_process(self):
        try:
            self.camera = Picamera2()
            print("PiCamera2 initialized successfully.")

        except RuntimeError as e:
            print(f"Error initializing PiCamera2: {e}")
            self.camera = None
            
        self.configure_camera()
        """Capture images in a separate process."""
        if self.camera is None:
            print("Camera not initialized. Exiting capture process.")
            return
        else:
            print("Starting camera capture process...")

        try:
            print("Capturing images...")
            while not self.stop_event.is_set():
                # frame = self.camera.capture_array()
                request = self.camera.capture_request()
                # timestamp = get_time()
                frame = request.make_array("main")
                request.release()
                print("[Picamera]:", get_time())
                with self.lock:
                    self.shared_frame[0] = frame.copy()
                    # self.shared_timestamp.value = timestamp
                # time.sleep(0.01)  # ~100 FPS

        except RuntimeError as e:
            print(f"Error during capture: {e}")

        finally:
            self.camera.stop()
            print("Camera capture stopped.")
            
    def preProcess(self, img):
        """Preprocess the image."""
        if img is None:
            print("No image to preprocess.")
            return None
        img = img[120:, :, :]  # Crop the bottom half
        img = cv2.resize(img, (320, 180))  # Resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        # print(f"Preprocessed image shape: {img.shape}")
        return img

    def stop(self):
        """Stops the image capture process."""
        self.stop_event.set()  # Signal the process to stop
        self.process.join()    # Wait for the process to finish
        print("ImageCapture process stopped.")

    def get_frame(self):
        """
        Returns the most recent frame captured by the camera.
        If no frame has been captured yet, returns None.
        """
        # with self.lock:
        frame = self.shared_frame[0].copy()
        # timestamp = self.shared_timestamp.value
        # remove offset from right
        frame = frame[:, :self.size[0]-self.offset, :]
        # return frame
        return self.preProcess(frame),frame
        # return None,frame

    def erase_frame(self):
        """Clears the shared frame."""
        with self.lock:
            # print("Erasing frame...")
            self.shared_frame[0] = None

if __name__ == "__main__":
    cam = ImageCapture(size = (1280, 960))
    cam.start_capturing()  # Start the capture process
    time.sleep(5) # Wait for the camera to initialize

    try:
        for idx in range(5):  # Capture 5 frames as a test
            tic = time.time()
            _,frame = cam.get_frame()
            print(f"Time to get frame: {time.time() - tic:.3f} s")
        cam.stop()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cam.stop()  # Stop the capture process
        print("Program exited cleanly.")
