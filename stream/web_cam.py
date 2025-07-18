#!/usr/bin/python3

import io
import logging
import socketserver
from threading import Condition
from http import server
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, Quality # This will be for encoding video
from picamera2.outputs import FfmpegOutput  # Use FfmpegOutput for H264 output

PAGE = """\
<html>
<head>
<title>Raspberry Pi - Surveillance Camera</title>
</head>
<body>
<center><h1>Raspberry Pi - Surveillance Camera</h1></center>
<center><img src="stream.mjpg" width="640" height="480"></center>
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

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

# Initialize Picamera2
with Picamera2() as camera:
    # Configure the camera settings (resolution and format)
    camera.video_configuration.controls.FrameRate = 10.0
    camera_config = camera.create_video_configuration({"size": (640, 480), "format": "YUV420"})
    camera.configure(camera_config)

    # Set the frame rate (FPS) to 10
    

    # Create a StreamingOutput to capture the video frames
    output = StreamingOutput()

    # Start preview (this is optional but good for ensuring the camera works)
    camera.start_preview(Preview.NULL)
    encoder = H264Encoder()
    output = FfmpegOutput("http://192.168.50.124:8888/stream.jpg")  # Use FfmpegOutput with file path
    encoder.audio = False
    # Start recording MJPEG stream using the StreamingOutput
    camera.start_recording(encoder,output)

    try:
        # Set up the server to stream MJPEG over HTTP
        address = ('192.168.50.124', 8888)
        server = StreamingServer(address, StreamingHandler)
        print("Server started at http://192.168.50.124:8888/")
        server.serve_forever()
    finally:
        # Stop recording when done
        camera.stop_recording()
