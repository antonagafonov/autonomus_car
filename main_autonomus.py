import threading
import time
from queue import Queue
from threading import Event
from ControllerModule import JoystickController
from utils import VehicleSteering, create_control_signal
from DataCollector import DataCollector
from Camera import ImageCapture
import os
import multiprocessing as mp
from image_inference_module import inference_loop
import numpy as np
import time
from datetime import datetime
from collections import deque

def main():
    # Create a queue for communication between Joystick and VehicleSteering threads
    input_queue = Queue()
    stop_event = Event()

    # Initilize Image_Inference class with shared memmory and flags
    # start shared_mem[0] = img, shared_mem[1] = speed, shared_mem[2] = steer, shared_mem[3] = prediction
    # in shape like this img.shape:  (1, 120, 360, 1) speed.shape:  (1, 30, 1) steer.shape:  (1, 30, 1) out.shape:  (1, 2, 10, 1)

    # Initialize Joystick and VehicleSteering classes
    joystick = JoystickController(input_queue=input_queue, stop_event=stop_event)
    joystick.start()
    vehicle_steering = VehicleSteering(input_queue=input_queue, stop_event=stop_event)
    vehicle_steering.start()

    # initialize camera
    camera = ImageCapture()
    camera.start_capturing()  # Start the capture thread

    # Initialize data collector
    data_collector = DataCollector()

    # Start the threads
    
    with mp.Manager() as manager:

        img_shape = (120, 360)
        speed_shape = (8,)
        steer_shape = (8,)
        prediction_shape = (2, 10)

        # Create zero-initialized NumPy arrays
        img_np = np.zeros(img_shape, dtype=np.float32)
        speed_np = np.zeros(speed_shape, dtype=np.float32)
        steer_np = np.zeros(steer_shape, dtype=np.float32)
        prediction_np = np.zeros(prediction_shape, dtype=np.float32)

        shared_dict = manager.dict()
        shared_dict["stop_event"] = False
        shared_dict["inference_in_progress"] = False
        shared_dict["run_inference"] = True
        shared_dict["infr_idx"] = 0
        shared_dict["img"] = img_np.flatten()
        shared_dict["speed"] = speed_np.flatten()
        shared_dict["steer"] = steer_np.flatten()
        shared_dict["pred"] = prediction_np.flatten()

        inference_process = mp.Process(target=inference_loop, args=(shared_dict,))
        inference_process.start()

        time.sleep(5)
        print("All processes are started.")

        # create two queues , one for speed history and one for steer history, each stores last 30 values
        speed_history = deque([0] * 8, maxlen=8) 
        steer_history = deque([0] * 8, maxlen=8) 

        # try:
        # Main loop
        control_idx = 0
        main_loop_delta = 0.15
        idx = 0
        prev_inf_idx = shared_dict["infr_idx"]
        print("Starting main loop with infr_idx:", shared_dict["infr_idx"])
        last_prediction = np.array(shared_dict["pred"]).reshape(2,10)

        while not stop_event.is_set():
            main_loop_start = time.time()

            # take prediction from shared_dict["pred"] this are 10 predictions for 0.15 sec each,
            # so we can take the first prediction and use it for the next 0.15*10 = 1.5 sec 
            # we are taking those until we see that shared_dict["infr_idx"] is changed, if changed
            # we take the next 10 predictions

            if shared_dict["infr_idx"] > prev_inf_idx: # new prediction is available
                control_idx = 0 # reset the control index
                prev_inf_idx = shared_dict["infr_idx"] # update the prev_inf_idx
                last_prediction = np.array(shared_dict["pred"]).reshape(2,10) # get the last prediction

            print("main loop idx:",idx,"prev_inf_idx:",prev_inf_idx,"infr idx:",shared_dict["infr_idx"],"control_idx: ",control_idx)

            state,control_idx = create_control_signal(last_prediction, control_idx,record = True)

            last_speed,last_steer = state["forward"],state["steering"]

            # add last_speed in the beginning of the queue
            speed_history.appendleft(last_speed)
            # add the last speed and steer to the history
            steer_history.appendleft(last_steer)

            # print("Speed history:",speed_history)
            # print("Steer history:",steer_history)

            # udpate shared_dict["speed"] and shared_dict["steer"]
            shared_dict["speed"] = np.array(speed_history).flatten()
            shared_dict["steer"] = np.array(steer_history).flatten()
            print("control_idx:",control_idx,"State:",state)
            im,_ = camera.get_frame()

            if im is not None:
                # starting inference loop
                # insert image to shared memmory shared_mem[0]
                shared_dict["img"] = im
                print("Image inserted to shared memmory:",shared_dict["infr_idx"])
                # print(f"Captured frame shape: {im.shape}")
                # print("Got frame, ereasing from shared memmory")
                # camera.erease_frame()
            else:
                print("No frame captured, skipping iteration.")
                continue
            
            # print("prediction: ",shared_dict["pred"])
            # Get the state from the joystick
            j_state = joystick.get()  # get_state() should return current joystick state 
            
            # Pass the state to the VehicleSteering class to control the vehicle
            if state:  # Check if there's a valid state
                # print(f"Joystick state: {state}")
                # Joystick state: {'steering': -1,1,0, 'forward': 1,0, 'backward': 1,0}
                if j_state.get("exit", 0) == 1:
                    shared_dict["run_inference"] = False
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
                # print("Speed: {}, Turn: {}".format(speed, turn))
                vehicle_steering.move(speed=speed, turn=-turn,boost = state.get('boost',0))
            if state["recording"] == 1:
                print("Recording data...")
                data_collector.saveData(im, state)

            main_loop_end = time.time()
            dt = main_loop_end - main_loop_start
            time.sleep(max(0,0.15-dt))  # Adjust the sleep time to control the loop frequency

            print(f"Main loop iteration time: {time.time() - main_loop_start} sec")
            idx += 1
                
        # except KeyboardInterrupt:
        #     # Graceful exit on keyboard interrupt (Ctrl+C)
        #     print("Stopping the program...")

        # finally:
        #     # Graceful cleanup
        #     print("Stopping all components...")
        #     data_collector.save_data_to_file()
        #     stop_event.set()  # Signal threads to stop

        #     camera.stop()  # Stop the camera thread
        #     print("Camera stopped.")

        #     inference_process.terminate()
        #     inference_process.join()

        #     vehicle_steering.exit_program()
        #     joystick.join()  # Wait for joystick thread to finish
        #     vehicle_steering.join()  # Wait for steering thread to finish
        #     print("Program exited cleanly.")
        #     os._exit(0)  # Force exit if necessary

if __name__ == '__main__':
    main()
