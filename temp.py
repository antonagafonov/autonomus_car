import time
import os
import cv2
import numpy as np
import multiprocessing as mp
import queue
from threading import Thread
from picamera2 import Picamera2, Metadata
from utils import get_time

os.environ["LIBCAMERA_LOG_LEVELS"] = "ERROR"  # Only log errors

class ImageCapture:
    def __init__(self, size=(640, 480), offset=30, save_dir="captured_images"):
        self.manager = mp.Manager()
        self.shared_frame = self.manager.list([None])
        self.shared_timestamp = self.manager.Value("d", 0.0)
        self.lock = mp.Lock()
        self.stop_event = self.manager.Event()
        self.size = size
        self.offset = offset

        # Image queue for saving
        self.task_queue = queue.Queue()
        self.save_thread = Thread(target=self.save_images_from_queue, daemon=True)
        
        # Image saving directory
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.csv_file = os.path.join(self.save_dir, "image_log.csv")

        # Process for capturing images
        self.process = mp.Process(target=self._capture_process)

    def start_capturing(self):
        """Start image capture process and saving thread."""
        self.process.start()
        self.save_thread.start()  # Start image saving thread

    def _capture_process(self):
        """Capture images and store them in shared memory and queue."""
        try:
            self.camera = Picamera2()
            print("PiCamera2 initialized successfully.")
        except RuntimeError as e:
            print(f"Error initializing PiCamera2: {e}")
            self.camera = None

        self.configure_camera()
        if self.camera is None:
            print("Camera not initialized. Exiting capture process.")
            return

        print("Capturing images...")
        try:
            idx = 0
            while not self.stop_event.is_set():
                request = self.camera.capture_request()
                frame = request.make_array("main")
                request.release()
                timestamp = get_time()

                with self.lock:
                    self.shared_frame[0] = frame.copy()

                # Add frame to queue with index
                self.task_queue.put((idx, frame, timestamp))
                idx += 1

        except RuntimeError as e:
            print(f"Error during capture: {e}")
        finally:
            self.camera.stop()
            print("Camera capture stopped.")

    def save_images_from_queue(self):
        """Continuously saves images from the queue."""
        with open(self.csv_file, "w") as f:
            f.write("index,timestamp,filename\n")  # CSV header

        while not self.stop_event.is_set() or not self.task_queue.empty():
            try:
                idx, frame, timestamp = self.task_queue.get(timeout=1)

                filename = os.path.join(self.save_dir, f"image_{idx:04d}.jpg")
                cv2.imwrite(filename, frame)

                # Log in CSV
                with open(self.csv_file, "a") as f:
                    f.write(f"{idx},{timestamp},{filename}\n")

            except queue.Empty:
                continue  # No new images, keep checking

    def stop(self):
        """Stops image capturing and saving."""
        self.stop_event.set()
        self.process.join()
        self.save_thread.join()
        print("ImageCapture stopped.")

if __name__ == "__main__":
    cam = ImageCapture(size=(1280, 960))
    cam.start_capturing()
    time.sleep(5)  # Allow time for images to be captured

    try:
        for _ in range(5):  # Retrieve some frames
            _, frame = cam.get_frame()
            time.sleep(1)
        cam.stop()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cam.stop()
        print("Program exited cleanly.")
