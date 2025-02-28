from picamera2 import Picamera2
import time
import threading
from datetime import datetime

# Initialize camera
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480)},
    # main={"size": (640, 480), "format": "YUV420"},
    controls={
        "FrameRate": 60,            # Target 90 FPS
        "ExposureTime": 5000,       # Reduce exposure to allow higher FPS
        "AnalogueGain": 1.0,        # Fix gain to prevent auto adjustments
        "AwbEnable": False,         # Disable Auto White Balance
        "AeEnable": False,          # Disable Auto Exposure
        "FrameDurationLimits": (5000, 11111),  # Min exposure to allow 90 FPS
    })
picam2.configure(config)


# Start camera capture
shared_frame = [None]
lock = threading.Lock()
stop_event = threading.Event()

def capture_loop():
    print("Capturing images...")
    picam2.start()  # Start camera before loop
    timestamps = []
    try:
        while not stop_event.is_set():
            # print picam2 config
            # metadata = picam2.capture_metadata()

            # frame = picam2.capture_array()
            request = picam2.capture_request()
            frame = request.make_array("main")
            request.release()
            with lock:
                shared_frame[0] = frame.copy()
            current_time = datetime.now()
            timestamps.append(current_time)
            # print("[Picamera]:", current_time.strftime("%H:%M:%S.%f"))
            # time.sleep(0.01)  # Prevent excessive CPU usage
    finally:
        print("Stopping camera...")
        picam2.stop()  # Stop camera safely
        # print(metadata)
        # Calculate deltas and mean
        deltas = [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0
        print(f"Mean delta between frames: {mean_delta:.6f} seconds")
        fps = 1 / mean_delta if mean_delta > 0 else 0
        print(f"Estimated FPS: {fps:.2f} fps")
# Start capture in a separate thread
capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()

# Run for 10 seconds then stop (for testing)
time.sleep(5)
stop_event.set()
capture_thread.join()  # Wait for thread to exit

# Ensure camera is stopped before exiting
picam2.stop()
picam2.close()
