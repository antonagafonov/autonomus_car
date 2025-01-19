import threading
import time
from queue import Queue
from threading import Event
from ControllerModule import JoystickController
from utils import VehicleSteering,calibrate_steering
from DataCollector import DataCollector
from Camera import ImageCapture
import os
from utils import get_time

# Assuming Joystick and VehicleSteering classes are imported from elsewhere
    
def main():
    # Create a queue for communication between Joystick and VehicleSteering threads
    input_queue = Queue()
    stop_event = Event()

    # Initialize Joystick and VehicleSteering classes
    joystick = JoystickController(input_queue=input_queue, stop_event=stop_event)
    joystick.start()
    vehicle_steering = VehicleSteering(input_queue=input_queue, stop_event=stop_event)
    vehicle_steering.start()

    camera = ImageCapture()
    camera.start_capturing()

    data_collector = DataCollector()

    # wait 2 seconds for the threads to start
    time.sleep(5)
    try:
        while not stop_event.is_set():
            # Retrieve the state from the joystick (you can define the joystick states as needed)
            # Here I assume 'state' is a dictionary with 'speed' and 'turn' as keys
            state = joystick.get()  # get_state() should return current joystick state
            print("State: ", state)
            _,im = camera.get_frame()
            print("Time:",get_time())
            if im is None:
                print("No frame captured, skipping iteration.")
                continue
            # Pass the state to the VehicleSteering class to control the vehicle
            if state:  # Check if there's a valid state
                # Joystick state: {'steering': -1,1,0, 'forward': 1,0, 'backward': 1,0}
                if state.get("exit", 0) == 1:
                    vehicle_steering.stop_motors()
                    print("Exiting the program...")
                    break
                if state["forward"] > 0 and state["backward"] == 0:
                    speed = state["forward"] 
                elif state["backward"] > 0 and state["forward"] == 0:
                    speed = -state["backward"]
                else:
                    speed = 0.0
                    vehicle_steering.stop_motors()
                turn = state.get('steering', 0)  # Default turn to 0 if not in state

                print("Speed: {}, Turn: {}".format(speed, turn))
                vehicle_steering.move(speed=speed, turn=-turn,boost = state.get('boost',0))
            if state["recording"] == 1:
                print("Recording data...")
                data_collector.saveData(im, state)
            time.sleep(0.05)  # Adjust the sleep time to control the loop frequency
    except KeyboardInterrupt:
        # Graceful exit on keyboard interrupt (Ctrl+C)
        print("Stopping the program...")

    finally:
        # Graceful cleanup
        data_collector.save_data_to_file()
        print("Stopping all components...")
        stop_event.set()  # Signal threads to stop
        vehicle_steering.exit_program()
        joystick.join()  # Wait for joystick thread to finish
        vehicle_steering.join()  # Wait for steering thread to finish
        print("Program exited cleanly.")
        os._exit(0)  # Force exit if necessary

if __name__ == '__main__':
    main()

