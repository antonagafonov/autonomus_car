import time
import os
import cv2
import atexit
import RPi.GPIO as GPIO
from queue import Queue
from threading import Event
import threading

from ControllerModule import JoystickController
from utils import VehicleSteering,save_images,calibrate_steering,get_deviation,PIDController,get_time
from DataCollector import DataCollector
from Camera import ImageCapture

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

    pid = PIDController()

    # wait 2 seconds for the threads to start
    time.sleep(5)
    m_idx = 0

    prev_enable_pid = 0
    loop_times = []
    try:
        while not stop_event.is_set():
            loop_start_time = time.time()

            # Retrieve the state from the joystick (you can define the joystick states as needed)
            # Here I assume 'state' is a dictionary with 'speed' and 'turn' as keys
            state = joystick.get()  # get_state() should return current joystick state
        
            # print("1. Time:",m_idx, get_time())
            _,im = camera.get_frame()
            # print("2. Time:",m_idx, get_time())
            len_contours,cte,countours_img,cutted_threshold,inversed = get_deviation(im)
            # print("3. Time:",m_idx, get_time())
            if cte is None:
                print("No deviation detected, skipping iteration.")
                cte = 0

            if not os.path.exists("/home/toon/data/temp_data"):
                os.makedirs("/home/toon/data/temp_data")
            # print("4. Time:",m_idx, get_time())
            steer_pid = pid.control(cte = cte, dt = 0.075)  # Calculate the steering angle using PID
            # print("5. Time:",m_idx, get_time())

            # Prepare the filenames
            img_filename = f"/home/toon/data/temp_data/img_{m_idx}_{cte}_{steer_pid}_{str(state['enable_pid'])}.png"
            countours_filename = f"/home/toon/data/temp_data/countours_img_{m_idx}_{cte}_{steer_pid}_{str(state['enable_pid'])}.png"
            cutted_threshold_filename = f"/home/toon/data/temp_data/cutted_threshold_{m_idx}_{cte}_{steer_pid}_{str(state['enable_pid'])}.png"
            # create dict with img_filename , countours_filename and cutted_threshold_filename as keys and images as values
            img_dict = {}
            # print("5.1. Time:",m_idx, get_time())
            img_dict[img_filename] = im
            img_dict[countours_filename] = countours_img
            img_dict[cutted_threshold_filename] = cutted_threshold
            # print("5.2. Time:",m_idx, get_time())
            # Create threads for saving images
            thread = threading.Thread(target=save_images, args=(img_dict,))
            # print("5.3. Time:",m_idx, get_time())
            # Start threads
            thread.start()
            # print("5.4. Time:",m_idx, get_time())
            # Optionally, join the threads to ensure they finish before moving on to the next iteration
            
            # print("5.5. Time:",m_idx, get_time())
            # print("6. Time:",m_idx, get_time())

            betha = 0.3
            if state["enable_pid"] == 1:
                prev_enable_pid = 1
                if abs(steer_pid) > 0.3 and abs(steer_pid) < 0.6:
                    state["forward"] = 0.3+betha
                elif abs(steer_pid) > 0.6:
                    state["forward"] = 0.15+betha
                else:
                    state["forward"] = 0.4+betha
                if len_contours == 1:
                    state["forward"] = 0.3+betha
                state["steering"] = steer_pid
            elif state["enable_pid"] == 0 and prev_enable_pid == 1:
                pid.reset()
                state["forward"] = 0
                state["steering"] = 0
                prev_enable_pid = 0
            else:
                pass

            # print("Steering Angle:",steer_pid)
  
            if im is None:
                print("No frame captured, skipping iteration.")
                continue
            # Pass the state to the VehicleSteering class to control the vehicle
            if state["recording"] == 1:
                print("Recording data...")
                data_collector.saveData(im, state)
            # print("8. Time:",m_idx, get_time())
            if state:  # Check if there's a valid state
                # Joystick state: {'steering': -1,1,0, 'forward': 1,0, 'backward': 1,0}
                if state.get("exit", 0) == 1:
                    vehicle_steering.stop_motors()
                    print("Exiting the program...")
                    break
                print("9. Time:",m_idx, get_time())
                if state["forward"] > 0 and state["backward"] == 0:
                    speed = state["forward"] 
                elif state["backward"] > 0 and state["forward"] == 0:
                    speed = -state["backward"]
                else:
                    speed = 0.0
                    vehicle_steering.stop_motors()
                # print("10. Time:",m_idx, get_time())
                turn = state.get('steering', 0)  # Default turn to 0 if not in state
                print("Speed: {}, Turn: {}".format(speed, turn))
                # print("10.1. Time:",m_idx, get_time())
                # self,speed=0.5,turn=0,boost = 0,t=0.05, steering_offset=0.0,s = 80
                vehicle_steering.move(speed=speed, turn=-turn,boost = state.get('boost',0),t=0.025)
                # print("11. Time:",m_idx,get_time())
            # time.sleep(0.05)  # Adjust the sleep time to control the loop frequency
            m_idx += 1
            loop_end_time = time.time()
            # print("12. Time:",m_idx, get_time())
            loop_time = loop_end_time - loop_start_time
            print("Loop time:", loop_time)
            loop_times.append(loop_time)
            if len(loop_times) > 10:
                print("Average loop time:", sum(loop_times) / len(loop_times))
    except KeyboardInterrupt:
        # Graceful exit on keyboard interrupt (Ctrl+C)
        print("Stopping the program...")

    finally:
        try:
            # Graceful cleanup
            data_collector.save_data_to_file()
            print("Stopping all components...")
            stop_event.set()  # Signal threads to stop
            vehicle_steering.exit_program()
            joystick.join()  # Wait for joystick thread to finish
            vehicle_steering.join()  # Wait for steering thread to finish

            # GPIO cleanup
            print("Cleaning up GPIO...")
            GPIO.cleanup()  # Clean up any GPIO settings
            print("Program exited cleanly.")

        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Force exit if necessary
            os._exit(0)  # Force exit if cleanup fails
if __name__ == '__main__':
    main()

