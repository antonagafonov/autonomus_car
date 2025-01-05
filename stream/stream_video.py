#!/usr/bin/python3

import socket
from threading import Event
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput  # Use FfmpegOutput instead
from picamera2.outputs import FileOutput

picam2 = Picamera2()
picam2.video_configuration.controls.FrameRate = 10.0
# Create video configuration with the desired size and format
video_config = picam2.create_video_configuration({"size": (1280, 720), 'format': 'YUV420'})

# Set the frame rate (FPS) separately using the camera's configure method
picam2.configure(video_config)


encoder = H264Encoder(bitrate=10000000)
encoder.audio = False

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("192.168.50.124", 8888))

    while True:
        print("Waiting")
        sock.listen()

        conn, addr = sock.accept()
        print(f"Connected from {addr}")

        # Create output using FfmpegOutput to handle H264 encoding and file saving
        output = FfmpegOutput("test.h264")  # Use FfmpegOutput with file path
        event = Event()
        output.error_callback = lambda _: event.set()  # noqa

        # Start recording with H264 encoder and the file output
        picam2.start_recording(encoder, output)

        event.wait()
        print("Disconnected")

        picam2.stop_recording()
