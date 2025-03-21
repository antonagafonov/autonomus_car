#!/usr/bin/python3

# Mostly copied from https://picamera.readthedocs.io/en/release-1.13/recipes2.html
# Run this script, then point a web browser at http:<this-ip-address>:8000
# Note: needs simplejpeg to be installed (pip3 install simplejpeg).

import io
import logging
import socketserver
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import cv2
import numpy as np


PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            # self.frame = buf
            # self.condition.notify_all()
            # Convert JPEG buffer to a NumPy array
            image_array = np.frombuffer(buf, dtype=np.uint8)

            frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            h, w, _ = frame.shape
            # print(frame.shape)
            frame = frame = frame[:, :w-30, :]
            frame = cv2.resize(frame, (320, 180))  # Resize
            # Get image dimensions
            h, w, _ = frame.shape
            # print(frame.shape)
            center_x, center_y = w // 2, h // 2  # Center point
            bottom_center = (w // 2, h)  # Bottom center of the image
            left_middle = (0, h // 2)  # Middle of the left side
            right_middle = (w, h // 2)  # Middle of the right side

            # Draw the center red vertical line
            cv2.line(frame, (center_x, 0), (center_x, h), (0, 0, 255), 2)

            # Draw two lines from bottom center to left and right middle
            cv2.line(frame, bottom_center, left_middle, (0, 255, 0), 2)
            cv2.line(frame, bottom_center, right_middle, (0, 255, 0), 2)

            # Encode frame back to JPEG format
            _, encoded_frame = cv2.imencode('.jpg', frame)
            self.frame = encoded_frame.tobytes()
            self.condition.notify_all()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def estimate_camera_matrix(width, height):
    cx, cy = width / 2, height / 2
    fx, fy = cx, cy  # Rough assumption
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K

def shift_camera_matrix_right(K, shift_value=100):
    # Modify the principal point (cx)
    K[0, 2] += shift_value  # Shifting cx to the right
    return K

picam2 = Picamera2()    
print(picam2.sensor_modes)



picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
output = StreamingOutput()
picam2.start_recording(JpegEncoder(), FileOutput(output))

try:
    address = ('', 8888)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()
finally:
    picam2.stop_recording()