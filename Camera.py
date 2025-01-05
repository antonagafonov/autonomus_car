import time
import os
import cv2  # For web camera support
import numpy as np
from picamera2 import Picamera2
import multiprocessing as mp

os.environ["LIBCAMERA_LOG_LEVELS"] = "ERROR"  # Only log errors


class ImageCapture:
    """
    Captures images using the PiCamera2 or a web camera in a separate process.
    The latest image is always stored in shared memory.
    """

    def __init__(self):

        # Shared memory and synchronization primitives
        self.manager = mp.Manager()
        self.shared_frame = self.manager.list([None])  # Initialize with None
        self.lock = mp.Lock()
        self.stop_event = self.manager.Event() 
        # Process for capturing images
        self.process = mp.Process(target=self._capture_process)

    def configure_camera(self):
            """Configures the PiCamera2 for image capture."""
            if self.camera:
                config = self.camera.create_still_configuration(main={"size": (640, 480)})
                # camera_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
                self.camera.configure(config)
                self.camera.start()
                time.sleep(2)  # Allow the camera to initialize
                print("Camera configured successfully.")
            else:
                print("Camera not available, cannot configure.")

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
                frame = self.camera.capture_array()
                with self.lock:
                    # print(f"Captured frame: {frame.shape} pushed with lock")
                    self.shared_frame[0] = frame
                time.sleep(0.01)  # ~100 FPS

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
        img = img[240:, :, :]  # Crop the bottom half
        img = cv2.resize(img, (360, 120))  # Resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # img = img / 255.0  # Normalize
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
        with self.lock:
            frame = self.shared_frame[0]
            return frame
            # return self.preProcess(frame),frame

    def erase_frame(self):
        """Clears the shared frame."""
        with self.lock:
            # print("Erasing frame...")
            self.shared_frame[0] = None

if __name__ == "__main__":
    cam = ImageCapture()
    cam.start_capturing()  # Start the capture process
    time.sleep(5) # Wait for the camera to initialize

    try:
        for idx in range(5):  # Capture 5 frames as a test
            frame = cam.get_frame()
            if frame is not None:
                # print(f"Captured frame shape: {frame.shape}")
                # save to file with idx name
                cv2.imwrite(f"frame_{idx}.png", frame)
                cam.erase_frame()
            else:
                print("Waiting for frame...")
            time.sleep(0.3)  # Simulate processing delay
        cam.stop()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cam.stop()  # Stop the capture process
        print("Program exited cleanly.")
