import time
import os
import cv2
import atexit
import RPi.GPIO as GPIO
from queue import Queue
from threading import Event
import threading
from collections import deque

from ControllerModule import JoystickController
from utils import VehicleSteering,save_images,calibrate_steering,get_deviation,PIDController,get_time,inject_noise
from DataCollector import DataCollector
from Camera import ImageCapture
import random

def main(dt = 0.020):
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

    # data_collector = DataCollector()
    # data_collector.start()  # Start the thread

    # pid = PIDController()

    # wait 2 seconds for the threads to start
    time.sleep(5)
    m_idx = 0

    prev_enable_pid = 0
    loop_times = []

    if not os.path.exists("/home/toon/data/temp_data"):
        os.makedirs("/home/toon/data/temp_data")
    
    try:
        while not stop_event.is_set():
            loop_start_time = time.time()

            # _,im = camera.get_frame()
            # print("1. Time:",m_idx, get_time())
            state = joystick.get()
            # print("Joystick state:", state)
            # print("2. Time:",m_idx, get_time())
            # print("2. Time:",m_idx, get_time())

            # len_contours,cte,countours_img,cutted_threshold = get_deviation(im)
            # print("3. Time:",m_idx, get_time())
            # if cte is None:
            #     print("No deviation detected, skipping iteration.")
            #     cte = 0

            # steer_pid, speed_pid = pid.control(cte = cte, dt = 0.055)  # Calculate the steering angle using PID
            # print("4. Time:",m_idx, get_time())

            # if state["enable_pid"] == 1:
            #     # write pid steer value enable joistic controll
            #     state["steering"] = steer_pid
            #     state["forward"] = speed_pid
            #     prev_enable_pid = 1
            #     if len_contours == 1:
            #         print("Reducing speed, one lane detected!")
            #         state["forward"] = 0.4
        
            # elif state["enable_pid"] == 0 and prev_enable_pid == 1:
            #     pid.reset()
            #     state["forward"] = 0
            #     state["steering"] = 0
            #     prev_enable_pid = 0
            # else:
            #     pass
                
            # Prepare the filenames
            # img_filename = f"/home/toon/data/temp_data/img_{m_idx}_{cte}_{steer_pid}_{str(state['enable_pid'])}.png"
            # countours_filename = f"/home/toon/data/temp_data/countours_img_idx_{m_idx}_cte_{cte}_st_{state['steering']}_fw_{state['forward']}_pid_{str(state['enable_pid'])}.png"
            # cutted_threshold_filename = f"/home/toon/data/temp_data/cutted_threshold_{m_idx}_{cte}_{steer_pid}_{str(state['enable_pid'])}.png"

            # img_dict = {}

            # img_dict[countours_filename] = countours_img
            # print("5. Time:",m_idx, get_time())
            # thread = threading.Thread(target=save_images, args=(img_dict,))
            # print("6. Time:",m_idx, get_time())
            # thread.start()
            # print("7. Time:",m_idx, get_time())
            # if im is None:
            #     print("No frame captured, skipping iteration.")
            #     continue

            if state:  # Check if there's a valid state
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
                
                vehicle_steering.move(speed=speed, turn=-turn,boost = state.get('boost',0),t=0.0) 
                # print("move. Time:",m_idx, get_time())
            m_idx += 1
            loop_end_time = time.time()
            loop_time = loop_end_time - loop_start_time
            # print("Loop time:", loop_time)

            state["loop_time"] = loop_time

            # if state["recording"] == 1:
            #     # print("8. Time:",m_idx, get_time())
            #     data_collector.saveData(im, state)
            # loop_times.append(loop_time)
            
            if loop_time < dt:
                time.sleep(dt - loop_time)
            print("Loop time with delta:", time.time() - loop_start_time)

            # if len(loop_times) > 10:
            #     print("Average loop time:", sum(loop_times) / len(loop_times))
    except KeyboardInterrupt:
        # Graceful exit on keyboard interrupt (Ctrl+C)
        print("Stopping the program...")

    finally:
        try:
            # Graceful cleanup
            # data_collector.save_data_to_file()
            # data_collector.stop()
            # print("Data collector stopped.")

            print("Stopping all components...")
            stop_event.set()  # Signal threads to stop
            vehicle_steering.exit_program()
            joystick.stop()
            vehicle_steering.join()  # Wait for steering thread to finish

            camera.stop()
            print("Camera capture stopped.")

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

