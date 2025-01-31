import pygame
from pygame.locals import *
import time
import threading
from queue import Queue
import time
from datetime import datetime

class JoystickController(threading.Thread):
    def __init__(self, input_queue=None, stop_event=None):
        """Initialize the joystick and motor control interface."""
        threading.Thread.__init__(self)
        pygame.init()
        pygame.joystick.init()

        super().__init__()
        self.stop_event = stop_event
        self.input_queue = input_queue if input_queue else Queue()
        self.joystick = self.initialize_controller()
        if not self.joystick:
            raise Exception("No joystick detected!")
        # Shared state dictionary for joystick status (steering, forward, backward)
        self.state = {
            "steering": 0,  # -1 (left), 0 (center), 1 (right)
            "forward": 0,   # 0 (off), 1 (on)
            "backward": 0,
            "boost": 0,   # 0 (off), 1 (on)
            "recording": 0,  # 0 (off), 1 (on)
            "exit": 0 ,      # 0 (off), 1 (on)
            "timestep": self.get_time(),
            "enable_pid": 0
        }

    def get_time(self):
        now = datetime.now()
        formatted_time = now.strftime("%d:%m:%Y:%H:%M:%S") + f":{now.microsecond // 1000:02d}"
        return formatted_time

    def initialize_controller(self):
        """Initialize the first joystick if available."""
        joystick_count = pygame.joystick.get_count()
        print(f"Detected {joystick_count} joystick(s)")
        
        if joystick_count == 0:
            print("No joystick detected!")
            return None
        
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Joystick detected: {joystick.get_name()}")
        return joystick
    
    def add_timestep(self):
        self.state["timestep"] = self.get_time()

    def process_inputs(self):
        """Poll controller inputs and process joystick events."""
        try:
            while not self.stop_event.is_set():
                for event in pygame.event.get():  # Process all events
                    if event.type == pygame.JOYBUTTONDOWN:
                        self.handle_button_press(event.button)
                    elif event.type == pygame.JOYBUTTONUP:
                        self.handle_button_release(event.button)

                    # Handle axis motion (e.g., Axis 0 for steering control)
                    if event.type == pygame.JOYAXISMOTION:
                        self.handle_axis_motion(event.axis, event.value)
                self.add_timestep()
                time.sleep(0.01)  # Limit polling rate
        except KeyboardInterrupt:
            print("\nExiting...")
            pygame.quit()

    def handle_button_press(self, button):
        """Handle joystick button press events."""
        if button == 1:  # Button 1 pressed (forward)
            # print("Button 1 pressed: Forward ON")
            self.state["forward"] = 1
        elif button == 0:  # Button 2 pressed (backward)
            # print("Button 0 pressed: Backward ON")
            self.state["backward"] = 1
        elif button == 8:
            print("Button 8 pressed: Exiting...")
            self.state["exit"] = 1
        elif button == 5:
            # print("Button 5 pressed: Boost...")
            self.state["boost"] = 1
        elif button == 4:
            if self.state["recording"] == 1:
                print("Button 4 pressed: Stop Recording...") 
                self.state["recording"] = 0
            else:
                print("Button 4 pressed: Start Recording...")
                self.state["recording"] = 1
        elif button == 9:
            if self.state["enable_pid"] == 1:
                print("Button 9 pressed: Disable PID...")
                self.state["enable_pid"] = 0
            else:
                print("Button 9 pressed: Enable PID...")
                self.state["enable_pid"] = 1
        
    def handle_button_release(self, button):
        """Handle joystick button release events."""
        if button == 1:  # Button 1 released (forward off)
            # print("Button 1 released: Forward OFF")
            self.state["forward"] = 0
        elif button == 0:  # Button 2 released (backward off)
            # print("Button 0 released: Backward OFF")
            self.state["backward"] = 0
        elif button == 5:
            # print("Button 5 released: Boost OFF")
            self.state["boost"] = 0

    def clip(self,value):
        if abs(value) < 0.01:
            return 0 
        return value
        
    def handle_axis_motion(self, axis, value):
        """Handle joystick axis movement."""
        if axis == 0:  # Axis 0 for steering control
            # print(f"Axis 0 moved to {value:.2f}")
            self.state["steering"] = round(self.clip(value),4)
        elif axis == 4: # Axis 4 for speed control
            # print(f"Axis 4 moved to {value:.2f}")
            if value < 0:
                self.state["forward"] = round(abs(self.clip(value)),4)
                self.state["backward"] = 0
            elif value > 0:
                self.state["backward"] = round(abs(self.clip(value)),4)
                self.state["forward"] = 0
            else:
                pass

    def run(self):
        """Start processing joystick inputs in a separate thread."""
        self.process_inputs()

    def stop(self):
        """Stop the joystick controller thread."""
        self.join()

    def get(self):  
        """Return the shared state dictionary."""
        return self.state
