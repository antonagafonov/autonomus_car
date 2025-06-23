# Autonomous Car Project

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)

## Overview

This project implements a real-time autonomous driving system using a Raspberry Pi 4, PiCamera2, and deep learning models. The system features end-to-end learning for steering and speed control, real-time inference, multi-threaded architecture, and comprehensive data collection capabilities for training and evaluation.

## Key Features

- **Real-time Autonomous Driving**: End-to-end neural network control with TensorFlow/Keras
- **Multi-threaded Architecture**: Separate processes for camera capture, inference, and motor control
- **Data Collection System**: Comprehensive logging of images, control signals, and timestamps
- **PID Lane Following**: Traditional computer vision-based lane following as backup
- **Joystick Control**: PlayStation/Xbox controller support for manual override and data collection
- **Real-time Inference**: Optimized inference pipeline with shared memory between processes
- **Modular Design**: Separate modules for camera, control, data collection, and inference

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera        │    │   Inference     │    │   Control       │
│   (PiCamera2)   │    │   (TensorFlow)  │    │   (GPIO/PWM)    │
│                 │    │                 │    │                 │
│ • Image Capture │───▶│ • Neural Net    │───▶│ • Motor Control │
│ • Preprocessing │    │ • Shared Memory │    │ • Servo Control │
│ • Data Logging  │    │ • Multi-Process │    │ • Safety Stop   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────────────────┐
                    │    Data Collection      │
                    │                         │
                    │ • Image Logging         │
                    │ • State Recording       │
                    │ • CSV Export           │
                    └─────────────────────────┘
```

## Hardware Requirements

### Core Components
- **Raspberry Pi 4** (4GB+ RAM recommended)
- **PiCamera2** or compatible CSI camera module
- **RC Car Chassis** with servo steering and ESC motor control
- **PlayStation/Xbox Controller** for manual control and data collection
- **MicroSD Card** (32GB+ Class 10)

### Electronic Components
- Servo motor for steering control
- ESC (Electronic Speed Controller) for motor control
- Jumper wires for GPIO connections
- Power supply (7.4V LiPo battery recommended)

### Optional Components
- Ultrasonic sensors for obstacle detection
- IMU for orientation tracking
- Additional cameras for stereo vision

## Software Dependencies

### Core Python Packages
```bash
# Computer Vision and Image Processing
pip install opencv-python
pip install picamera2
pip install numpy

# Deep Learning
pip install tensorflow
pip install keras

# Hardware Control
pip install RPi.GPIO
pip install pygame  # For joystick support

# Data Management
pip install pandas
pip install csv
```

### System Requirements
```bash
# Raspberry Pi OS setup
sudo apt update
sudo apt upgrade
sudo apt install python3-pip
sudo apt install libcamera-apps
sudo apt install python3-libcamera
```

## Installation and Setup

### 1. Clone Repository
```bash
git clone https://github.com/antonagafonov/autonomus_car.git
cd autonomus_car
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Hardware Setup
- Connect PiCamera2 to CSI port
- Wire servo motor to GPIO pin (configured in utils.py)
- Connect ESC to GPIO pin for motor control
- Ensure proper power supply connections

### 4. Camera Configuration
```bash
# Test camera functionality
python Camera.py
```

### 5. Controller Setup
```bash
# Test joystick connectivity
python ControllerModule.py
```

## Usage

### Manual Control Mode
```bash
# Basic manual control with data collection
python main_controller.py
```

**Controller Mapping:**
- **Left Stick**: Steering control
- **Right Trigger**: Forward acceleration
- **Left Trigger**: Reverse/brake
- **Button A**: Forward movement
- **Button B**: Backward movement
- **Button X**: Start/stop recording
- **Button Y**: Enable/disable PID mode
- **Select Button**: Exit program

### PID Lane Following Mode
```bash
# Computer vision-based lane following
python main_PID_lane_follow.py
```

This mode uses traditional computer vision techniques for lane detection and PID control for steering.

### Full Autonomous Mode
```bash
# Neural network-based autonomous driving
python main_autonomus.py
```

This mode uses the trained deep learning model for end-to-end autonomous driving.

### Autonomous Mode (No Joystick)
```bash
# Fully autonomous mode without manual override
python main_autonomus_wo_joystic.py
```

### Data Collection Only
```bash
# Collect data without motor control
python main_separate_data_grabbing.py
```

## Project Structure

```
autonomus_car/
├── Camera.py                    # PiCamera2 interface and image processing
├── ControllerModule.py          # PlayStation/Xbox controller interface
├── DataCollector.py             # Data logging and CSV export
├── image_inference_module.py    # TensorFlow inference engine
├── image_inference_module_class.py  # Object-oriented inference class
├── utils.py                     # Motor control and utility functions
├── main_controller.py           # Manual control main loop
├── main_PID_lane_follow.py      # PID-based lane following
├── main_autonomus.py            # Full autonomous driving
├── main_autonomus_wo_joystic.py # Autonomous without manual override
├── main_separate_data_grabbing.py # Data collection mode
├── models/                      # Trained neural network models
│   └── car_model_epoch_XX.keras
├── data/                        # Collected training data
│   ├── saved_images/
│   ├── image_log.csv
│   ├── steering_data.csv
│   └── data.txt
└── requirements.txt
```

## Key Modules

### Camera Module (`Camera.py`)
- **Multi-process image capture** using PiCamera2
- **Real-time preprocessing** (cropping, resizing, normalization)
- **Automatic image saving** with timestamp logging
- **Configurable frame rates** and resolutions

```python
camera = ImageCapture(size=(640, 480))
camera.start_capturing()
preprocessed_frame, raw_frame = camera.get_frame()
```

### Controller Module (`ControllerModule.py`)
- **Real-time joystick input processing**
- **Thread-safe state management**
- **Automatic data logging** of control inputs
- **Configurable button mappings**

```python
joystick = JoystickController(stop_event=stop_event)
joystick.start()
state = joystick.get()  # Get current control state
```

### Data Collector (`DataCollector.py`)
- **Asynchronous data saving** to prevent blocking
- **Synchronized image and state logging**
- **CSV export functionality**
- **Thread-safe queue management**

```python
data_collector = DataCollector(output_dir="data")
data_collector.start()
data_collector.saveData(image, control_state)
```

### Inference Module (`image_inference_module.py`)
- **Multi-process TensorFlow inference**
- **Shared memory optimization** for real-time performance
- **History-based prediction** using previous control inputs
- **Optimized model loading** and prediction pipeline

## Neural Network Architecture

The system uses a multi-input neural network that processes:
- **Current camera image** (180×320 grayscale)
- **Speed history** (last 8 speed values)
- **Steering history** (last 8 steering values)

**Output**: 10 future predictions (steering and speed) for 1.5 seconds ahead

### Model Input Shapes
```python
image_input: (1, 180, 320, 1)      # Preprocessed camera frame
speed_input: (1, 8, 1)             # Speed history
steer_input: (1, 8, 1)             # Steering history
```

### Model Output Shape
```python
prediction: (1, 2, 10, 1)          # 10 future [steering, speed] pairs
```

## Data Collection

### Training Data Format
The system automatically collects synchronized data:

**Image Log (`image_log.csv`)**:
```csv
index,timestamp,filename
000000,1640995200.123,image_000000.jpg
000001,1640995200.173,image_000001.jpg
```

**Control Log (`steering_data.csv`)**:
```csv
idx,steering,forward,backward,boost,recording,timestamp
0,-0.2345,0.6,0,0,1,1640995200.123
1,-0.1234,0.65,0,0,1,1640995200.173
```

### Data Collection Best Practices
1. **Collect diverse scenarios**: Different lighting, tracks, speeds
2. **Balanced steering data**: Equal left/right turn examples
3. **Smooth control inputs**: Avoid jerky movements
4. **Consistent timing**: Maintain steady data collection frequency
5. **Quality control**: Review collected data before training

## Performance Optimization

### Real-time Performance
- **Multi-processing architecture** prevents blocking
- **Shared memory** for efficient data transfer
- **Optimized image preprocessing** pipeline
- **Model quantization** for faster inference

### Memory Management
- **Circular buffers** for history tracking
- **Efficient image storage** with automatic cleanup
- **Memory-mapped arrays** for inter-process communication

### Typical Performance Metrics
- **Camera capture**: 30 FPS
- **Inference frequency**: 6-7 Hz
- **Control loop**: 10-20 Hz
- **Total system latency**: <150ms

## Troubleshooting

### Common Issues

**1. Camera Not Detected**
```bash
# Check camera connection
vcgencmd get_camera
# Test with libcamera
libcamera-still -o test.jpg
```

**2. Controller Not Recognized**
```bash
# List available joysticks
ls /dev/input/js*
# Test controller
jstest /dev/input/js0
```

**3. GPIO Permission Errors**
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER
# Reboot required
sudo reboot
```

**4. TensorFlow Model Loading Issues**
```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"
# Verify model path in inference module
```

**5. Performance Issues**
- Reduce camera resolution in `Camera.py`
- Lower inference frequency in main loops
- Enable GPU acceleration if available
- Check for thermal throttling

### Debug Mode
```bash
# Enable verbose logging
export LIBCAMERA_LOG_LEVELS="INFO"
python main_controller.py
```

## Safety Features

### Built-in Safety Mechanisms
- **Automatic motor stop** on controller disconnect
- **Emergency stop button** (controller select button)
- **Maximum speed limiting** in code
- **Watchdog timer** for inference process
- **Graceful shutdown** on interrupt

### Safety Guidelines
⚠️ **Important Safety Notes**:
- Always test in controlled environment
- Keep manual override ready
- Never test on public roads
- Ensure proper battery voltage monitoring
- Have physical kill switch accessible

## Training Your Own Model

### Data Preparation
1. Collect training data using `main_controller.py`
2. Organize data in proper directory structure
3. Preprocess images and normalize control inputs
4. Split data into training/validation sets

### Model Training
```python
# Example training script structure
model = create_model()
model.compile(optimizer='adam', loss='mse')
model.fit([images, speed_history, steer_history], 
          [future_controls], 
          epochs=100, 
          batch_size=32)
```

### Model Deployment
1. Save trained model as `.keras` format
2. Update model path in `image_inference_module.py`
3. Test with validation data before deployment
4. Gradually increase autonomy level during testing

## Contributing

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Test on actual hardware before submitting
- Update documentation for new features

### Testing Protocol
1. Unit tests for individual modules
2. Integration tests for multi-process communication
3. Hardware-in-the-loop testing
4. Safety validation in controlled environment

## Future Enhancements

- [ ] **Object Detection Integration**: YOLO-based obstacle detection
- [ ] **Stereo Vision**: Depth perception using dual cameras
- [ ] **SLAM Implementation**: Simultaneous localization and mapping
- [ ] **Advanced Control**: Model Predictive Control (MPC)
- [ ] **Sensor Fusion**: IMU and GPS integration
- [ ] **Real-time Visualization**: Web-based monitoring dashboard
- [ ] **Edge Optimization**: TensorRT acceleration
- [ ] **Multi-agent Coordination**: Vehicle-to-vehicle communication

## Technical Specifications

### System Requirements
- **Raspberry Pi 4** (4GB RAM minimum, 8GB recommended)
- **Python 3.8+**
- **TensorFlow 2.x**
- **OpenCV 4.x**
- **PiCamera2 library**

### Performance Benchmarks
- **Image processing**: 30 FPS @ 640×480
- **Neural network inference**: 6-7 Hz
- **End-to-end latency**: 100-150ms
- **Data logging**: 1000+ samples/minute

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Raspberry Pi Foundation for excellent hardware platform
- TensorFlow team for deep learning framework
- OpenCV community for computer vision tools
- PiCamera2 developers for camera interface

## Contact

**Anton Agafonov**
- GitHub: [@antonagafonov](https://github.com/antonagafonov)
- Project: [autonomus_car](https://github.com/antonagafonov/autonomus_car)

## References

- [End-to-End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- [PiCamera2 Documentation](https://datasheets.raspberrypi.org/camera/picamera2-manual.pdf)
- [TensorFlow Raspberry Pi Guide](https://www.tensorflow.org/lite/guide/python)
- [Raspberry Pi GPIO Documentation](https://www.raspberrypi.org/documentation/hardware/raspberrypi/gpio/)

---

**⚠️ Safety Notice**: This is an experimental autonomous vehicle system intended for educational and research purposes only. Always ensure proper safety measures are in place during testing. Never operate on public roads without proper authorization and safety equipment. The authors are not responsible for any damage or injury resulting from the use of this system.
