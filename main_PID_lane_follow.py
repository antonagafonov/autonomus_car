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
    data_collector.start()  # Start the thread

    pid = PIDController()

    # wait 2 seconds for the threads to start
    time.sleep(5)
    m_idx = 0

    prev_enable_pid = 0
    loop_times = []

    steering_queue = deque(maxlen=20)  # Optional max length to limit size
    inject_noise = False
    noised = 0
    if not os.path.exists("/home/toon/data/temp_data"):
        os.makedirs("/home/toon/data/temp_data")

    try:
        while not stop_event.is_set():
            loop_start_time = time.time()

            state = joystick.get()  # get_state() should return current joystick state
        
            _,im = camera.get_frame()

            len_contours,cte,countours_img,cutted_threshold = get_deviation(im)

            if cte is None:
                print("No deviation detected, skipping iteration.")
                cte = 0

            steer_pid = pid.control(cte = cte, dt = 0.055)  # Calculate the steering angle using PID

            # # Add steer_pid to the queue
            # steering_queue.append(float(steer_pid))
            # if inject_noise:
            #     if state["enable_pid"] == 1:
            #         steer_pid,noised = inject_noise(steering_queue)
            #         if noised:
            #             steering_queue.clear()
            #         print("steer_pid,noised:",steer_pid,noised)
            #     else:
            #         steering_queue.clear()
            #         noised = False
                     
            betha = 0.3
            if state["enable_pid"] == 1:
                # write pid steer value enable joistic controll
                # if abs(state["steering"]) < 0.05:
                state["steering"] = float(steer_pid)

                prev_enable_pid = 1
                if abs(steer_pid) > 0.3 and abs(steer_pid) < 0.6:
                    state["forward"] = 0.3+betha
                elif abs(steer_pid) > 0.6:
                    state["forward"] = 0.15+betha
                else:
                    state["forward"] = 0.4+betha
                if len_contours == 1:
                    state["forward"] = 0.3+betha
        
            elif state["enable_pid"] == 0 and prev_enable_pid == 1:
                pid.reset()
                state["forward"] = 0
                state["steering"] = 0
                prev_enable_pid = 0
            else:
                pass
                
            # Prepare the filenames
            # img_filename = f"/home/toon/data/temp_data/img_{m_idx}_{cte}_{steer_pid}_{str(state['enable_pid'])}.png"
            countours_filename = f"/home/toon/data/temp_data/countours_img_idx_{m_idx}_cte_{cte}_st_{state['steering']}_fw_{state['forward'] }_pid_{str(state['enable_pid'])}.png"
            # cutted_threshold_filename = f"/home/toon/data/temp_data/cutted_threshold_{m_idx}_{cte}_{steer_pid}_{str(state['enable_pid'])}.png"

            img_dict = {}

            img_dict[countours_filename] = countours_img

            thread = threading.Thread(target=save_images, args=(img_dict,))

            thread.start()

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
                vehicle_steering.move(speed=speed, turn=-turn,boost = state.get('boost',0),t=0.01)
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
            data_collector.stop()
            print("Data collector stopped.")

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

