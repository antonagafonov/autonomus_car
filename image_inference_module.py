# Load model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import time
import multiprocessing as mp
import cv2

def inference_loop(shared_dict = None):
    model_path = "/home/toon/scripts/models/tf_checkpoints_two_lanes_10_8/car_model_epoch_40.keras"
    # Load model
    custom_objects = {'mse': MeanSquaredError()}
    start_time = time.time()
    model = load_model(model_path, custom_objects=custom_objects)
    print("Model loaded,Time taken to load model:", time.time()-start_time,'sec')

    inf_idx = 0
    while shared_dict["run_inference"]:
        # Inference if inf flag is set
        if shared_dict["inference_in_progress"] == False:

            shared_dict["inference_in_progress"] == True # starting inference, so set flag to True
            
            if shared_dict["img"] is not None:
                print("Inference idx: ", inf_idx)

                img = shared_dict["img"]
                speed = shared_dict["speed"]
                steer = shared_dict["steer"]
                
                # convert to numpy array
                img = np.array(img).reshape((1,180,320,1))
                speed = np.array(speed).reshape((1,1,8,1))
                steer = np.array(steer).reshape((1,1,8,1))

                start = time.time()
                prediction = model.predict([img, speed, steer])
                print("Prediction: ", prediction)
                shared_dict["infr_idx"] += 1
                print("Time taken to predict:", time.time()-start,'sec')
                shared_dict["pred"] = prediction
                shared_dict["inference_in_progress"] == False # inference done, set flag to False
                inf_idx += 1
        else:
            # no inference, wat for 5ms
            time.sleep(0.005)

    
