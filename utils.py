import threading
import RPi.GPIO as GPIO
from time import sleep
import lgpio
import numpy as np
from datetime import datetime

def get_time():
    now = datetime.now()
    formatted_time = now.strftime("%d:%m:%Y:%H:%M:%S") + f":{now.microsecond // 1000:02d}"
    return formatted_time

def calibrate_steering(turn,max_turn = 0.8):
    if turn > 0:
        return max(turn-(1-max_turn),0)
    elif turn < 0:
        return min(turn+(1-max_turn),0)
    else:
        return 0
    
def get_time():
    now = datetime.now()
    formatted_time = now.strftime("%d:%m:%Y:%H:%M:%S") + f":{now.microsecond // 1000:02d}"
    return formatted_time

def create_control_signal(prediction, idx, record = False):
    if record:
        r = 1
    else:
        r = 0
    curr = [0.0,0.0]
    if idx>=10: # if we have reached the end of the prediction, stop the vehicle
        fw = 0.0
        bw = 0.0
        curr[0] = 0.0
        
    else:
        # Create a control signal based on the prediction
        curr = prediction[:, idx]
        print("curr: ",curr)
        if curr[1]>0.2:
            fw = curr[1]
            bw = 0.0
        elif curr[1]<-0.2:
            fw = 0.0
            bw = curr[1]
        else:
            fw = 0.0
            bw = 0.0

    state = {
    "steering": round(float(curr[0]),4),  # -1 (left), 0 (center), 1 (right)
    "forward": round(float(fw),4),   # 0 (off), 1 (on)
    "backward": bw,
    "boost": 0,   # 0 (off), 1 (on)
    "recording": 1,  # 0 (off), 1 (on)
    "exit": 0 ,      # 0 (off), 1 (on)
    "timestep": get_time()
            }
    
    return state, idx+1

class VehicleSteering(threading.Thread):
    def __init__(self, input_queue, stop_event):
        """
        Initialize the VehicleSteering thread.
        :param input_queue: A queue to receive input commands.
        :param stop_event: An event to signal stopping the thread.
        """
        super().__init__()
        self.input_queue = input_queue
        self.stop_event = stop_event

        self.cleanup_done = False  # Add a flag to track cleanup
        self.cleanup_lock = threading.Lock()  # Lock to synchronize cleanup

        # Motor Pins
        self.in1a = 24
        self.in2a = 23
        self.en_a = 25
        self.in1b = 20
        self.in2b = 16
        self.en_b = 21

        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([self.in1a, self.in2a, self.en_a, self.in1b, self.in2b, self.en_b], GPIO.OUT)
        GPIO.output([self.in1a, self.in2a, self.in1b, self.in2b], GPIO.LOW)

        # Initialize PWM
        self.pwm_a = GPIO.PWM(self.en_a, 100)
        self.pwm_b = GPIO.PWM(self.en_b, 100)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

        self.startup = True
        self.mySpeed = 0
        self.straight_count = 0

    def run(self):
        """Main loop to handle motor commands."""
        while not self.stop_event.is_set():
            try:
                key = self.input_queue.get(timeout=0.05)

                if self.startup:
                    self.stop_motors()
                    self.startup = False

            except Exception:
                pass

    def move(self,speed=0.5,turn=0,boost = 0,t=0.05, steering_offset=0.0,s = 80):
        if abs(turn) < 0.1:
            self.straight_count += 1
        else:
            self.straight_count = 0
        # inject noise to the steering
        # if self.straight_count > 10:
        #     print("injecting noise!")
        #     # inject random steering between 0.3 to 0.5 
        #     turn = np.random.uniform(0.9,1)
        #     # sample 1 or -1 for left or right
        #     turn = turn * np.random.choice([-1,1])
        #     self.straight_count = 0
        # Apply steering offset
        if speed > 0.05:
            if abs(turn) < 0.1:
                turn += steering_offset
        speed = round(speed * (s + (100-s) * boost),1)  # Boost increases speed by 20%
        turn = round(turn * 100,1)

        # print("moving with speed: ",speed," turn: ",turn)
        leftSpeed = speed-turn
        rightSpeed = speed+turn

        if leftSpeed>100:
            leftSpeed =100

        elif leftSpeed<-100:
            leftSpeed = -100

        if rightSpeed>100:
            rightSpeed =100

        elif rightSpeed<-100:
            rightSpeed = -100

        #print(leftSpeed,rightSpeed)
        self.pwm_a.ChangeDutyCycle(abs(leftSpeed))
        self.pwm_b.ChangeDutyCycle(abs(rightSpeed))

        if leftSpeed>0:
            GPIO.output(self.in1a,GPIO.HIGH)
            GPIO.output(self.in2a,GPIO.LOW)
        elif leftSpeed < 0:
            GPIO.output(self.in1a,GPIO.LOW)
            GPIO.output(self.in2a,GPIO.HIGH)
        else:
            GPIO.output(self.in1a, GPIO.LOW)
            GPIO.output(self.in2a, GPIO.LOW)  # Stop if speed is zero

        if rightSpeed>0:
            GPIO.output(self.in1b,GPIO.HIGH)
            GPIO.output(self.in2b,GPIO.LOW)
        elif rightSpeed < 0:
            GPIO.output(self.in1b,GPIO.LOW)
            GPIO.output(self.in2b,GPIO.HIGH)
        else:
            GPIO.output(self.in1b, GPIO.LOW)
            GPIO.output(self.in2b, GPIO.LOW)  # Stop if speed is zero
        sleep(t)

    def stop_motors(self, t=0):
        """Stop the motors."""
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)
        self.mySpeed = 0
        sleep(t)

    def set_speed(self, speed):
        """Set the current speed value."""
        self.mySpeed = speed

    def stop(self):
        with self.cleanup_lock:
            if not self.cleanup_done:

                try:
                    print("Stopping PWM and cleaning up...")
                    GPIO.cleanup()  # Clean up GPIO resources
                except Exception as e:
                    print(f"cleanup error: {e}")
                self.cleanup_done = True

    def exit_program(self):
        """Exit the program and clean up."""
        self.stop()
        self.stop_event.set()
        print("Exiting...")
