import time
import os
import cv2  # For web camera support
import numpy as np
import multiprocessing as mp
from picamera2 import Picamera2, Metadata
from utils import get_time
import queue
from threading import Thread
os.environ["LIBCAMERA_LOG_LEVELS"] = "ERROR"  # Only log errors
from copy import deepcopy

class ImageCapture:
    """
    Captures images using the PiCamera2 or a web camera in a separate process.
    The latest image is always stored in shared memory.
    """

    def __init__(self, size=(640, 480),offset = 30,save_dir = "data"):

        # Shared memory and synchronization primitives
        self.manager = mp.Manager()
        self.shared_frame = self.manager.list([None])  # Initialize with None
        self.shared_timestamp = self.manager.Value("d", 0.0)  # Shared memory for timestamp
        self.lock = mp.Lock()
        self.stop_event = self.manager.Event() 
        self.process = mp.Process(target=self._capture_process)

        self.size = size
        self.offset = offset
        self.task_queue = mp.Queue()  # Use multiprocessing.Queue

        self.image_idx = 0

        # Image saving directory
        self.save_dir = save_dir
        self.img_save_dir = os.path.join(self.save_dir, "saved_images")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.img_save_dir, exist_ok=True)
        self.csv_file = os.path.join(self.save_dir, "image_log.csv")

        # Start background thread for saving images
        self.saving_thread = Thread(target=self.save_images_from_queue, daemon=True)
        self.saving_thread.start()

    def configure_camera(self):
            """Configures the PiCamera2 for image capture."""
            if self.camera:
                config = self.camera.create_video_configuration(
                main={"size": (640, 480)},
                # main={"size": (640, 480), "format": "YUV420"},
                controls={
                    "FrameRate": 30,            # Target 90 FPS
                    "ExposureTime": 20000,       # Reduce exposure to allow higher FPS
                    "AnalogueGain": 1.0,        # Fix gain to prevent auto adjustments
                    "AwbEnable": True,         # Disable Auto White Balance
                    "AeEnable": True,          # Disable Auto Exposure
                    "FrameDurationLimits": (5000, 30000),  # Min exposure to allow 90 FPS
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
            return
        
        self.configure_camera()

        if self.camera is None:
            print("Camera not initialized. Exiting capture process.")
            return
        
        print("Starting camera capture process...")

        try:
            print("Capturing images...")
            while not self.stop_event.is_set():
                request = self.camera.capture_request()
                frame = request.make_array("main")
                request.release()

                with self.lock:
                    self.shared_frame[0] = frame.copy()

                # Put frame into task queue for saving
                self.task_queue.put((self.image_idx, self.shared_frame[0], get_time()))
                self.image_idx += 1  # Increment image index
                print("Captured image:", self.image_idx)

        except RuntimeError as e:
            print(f"Error during capture: {e}")

        finally:
            self.camera.stop()
            print("Camera capture stopped.")

    def save_images_from_queue(self):
        """Save images from the queue to disk."""
        with open(self.csv_file, "w") as f:
            f.write("index,timestamp,filename\n")  # CSV header
        # print(self.task_queue)
        while not self.stop_event.is_set():
            # print("Checking queue...")
            try:
                idx, frame, timestamp = self.task_queue.get(timeout=1)
                # print("Saving image:", idx)
                filename = os.path.join(self.img_save_dir, f"image_{idx:06d}.jpg")
                # save as rgb with cv2
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # save as bgr with picamera
                with open(self.csv_file,"a") as f:
                    f.write(f"{idx},{timestamp},{filename}\n")
            except queue.Empty:
                # print("Queue is empty.")
                continue # Skip if queue is empty
            # print("Saved image:", idx)

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
        # wait for saving thread to finish than stop, add while that checks if queue is empty with 1 sec sleep
        while not self.task_queue.empty():
            time.sleep(1)
        
        self.saving_thread.join()
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
    cam = ImageCapture(size = (640, 480))
    cam.start_capturing()  # Start the capture process
    time.sleep(2) # Wait for the camera to initialize
    cam.stop()  # Stop the capture process
    print("Program exited cleanly.")

    # try:
    #     for idx in range(5):  # Capture 5 frames as a test
    #         tic = time.time()
    #         _,frame = cam.get_frame()
    #         print(f"Time to get frame: {time.time() - tic:.3f} s")
    #     cam.stop()
    # except KeyboardInterrupt:
    #     print("Interrupted by user.")
    # finally:
    #     cam.stop()  # Stop the capture process
    #     print("Program exited cleanly.")
