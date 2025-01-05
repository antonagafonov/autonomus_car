# Load model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import time
import multiprocessing as mp
import cv2

# here we define inference calss that runs as separate process,
# it has shared memmory with main process and can be used to run inference on images and previus 30 actions of steering and speed
# the class load and initiates tf model and runs inference on images when gets the image from main process

def initialize_shared_mem():
    # Define the shapes
    img_shape = (120, 360)
    speed_shape = (30,)
    steer_shape = (30,)
    prediction_shape = (2, 10)

    # Create zero-initialized NumPy arrays
    img_np = np.zeros(img_shape, dtype=np.float32)
    speed_np = np.zeros(speed_shape, dtype=np.float32)
    steer_np = np.zeros(steer_shape, dtype=np.float32)
    prediction_np = np.zeros(prediction_shape, dtype=np.float32)

    # Flatten the arrays and transfer to mp.Array
    img = mp.Array('f', img_np.flatten())
    speed = mp.Array('f', speed_np.flatten())
    steer = mp.Array('f', steer_np.flatten())
    prediction = mp.Array('f', prediction_np.flatten())

    # Shared memory list for easy access
    shared_mem = {
        "img": img,
        "speed": speed,
        "steer": steer,
        "prediction": prediction,
    }

    shared_inf_run_flag = mp.Array('b', [True])
    shared_ready_to_inf_flag = mp.Array('b', [True])
    return shared_mem, shared_inf_run_flag, shared_ready_to_inf_flag


class Image_Inference(mp.Process):
    def __init__(self, shared_mem,shared_inf_run_flag,shared_ready_to_inf_flag, model_path = "/home/toon/scripts/models/car_model_epoch_85.keras"):

        super(Image_Inference, self).__init__()
        self.shared_mem = shared_mem
        self.shared_inf_run_flag = shared_inf_run_flag
        self.shared_ready_to_inf_flag = shared_ready_to_inf_flag
        self.model_path = model_path
        self.model = None

    def run(self):
        # Load model
        custom_objects = {'mse': MeanSquaredError()}
        start_time = time.time()
        self.model = load_model(self.model_path, custom_objects=custom_objects)
        print("Model loaded,Time taken to load model:", time.time()-start_time,'sec')

        while self.shared_inf_run_flag[0]:
            # Inference if inf flag is set
            if self.shared_ready_to_inf_flag[0] == True:
                self.shared_ready_to_inf_flag[0] = False # starting inference, so set ready to inf flag to false
                
                if self.shared_mem["img"] is not None:
                    # img.shape:  (1, 120, 360, 1) speed.shape:  (1, 30, 1) steer.shape:  (1, 30, 1) out.shape:  (1, 2, 10, 1)
                    img = self.shared_mem["img"]
                    speed = self.shared_mem["speed"]
                    steer = self.shared_mem["steer"]
                    
                    # convert to numpy array
                    img = np.array(img[:]).reshape((1,120,360,1))
                    speed = np.array(speed[:]).reshape((1,30,1))
                    steer = np.array(steer[:]).reshape((1,30,1))

                    # img = np.expand_dims(img, axis=0)
                    # speed = np.expand_dims(speed, axis=0)
                    # steer = np.expand_dims(steer, axis=0)
                    # print("img.shape: ", img.shape, "speed.shape: ", speed.shape, "steer.shape: ", steer.shape)
                    prediction = self.model.predict([img, speed, steer])
                    self.shared_mem[3] = prediction
                    self.shared_ready_to_inf_flag[0] = True # inference done, set ready to inf flag to true
            else:
                # no inference, wat for 10ms
                time.sleep(0.01)

        
    def stop(self):
        """Stop the inference process."""
        self.shared_inf_run_flag[0] = False  # Signal the loop to terminate
        self.join()  # Wait for the process to finish
        print("Inference process has been stopped.")
    
    def start_inference(self):
        self.shared_ready_to_inf_flag[0] = True