# Load model
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import time

# Map 'mse' to its explicit implementation
custom_objects = {'mse': MeanSquaredError()}
start_time = time.time()
model = load_model("/home/toon/scripts/models/car_model_epoch_85.keras", custom_objects=custom_objects)
print("Time taken to load model:", time.time()-start_time,'sec')
input("Press Enter to continue...")

print(model.summary())
input("Press Enter to continue...")
# load F:\AA-Private\car\scripts\models\test_data.npz
# and unpack to (imgBatch, speedInBatch, steerInBatch), axisOutBatch 

data = np.load("/home/toon/scripts/models/test_data.npz")
imgBatch = data['imgBatch']
speedInBatch = data['speedInBatch']
steerInBatch = data['steerInBatch']
axisOutBatch = data['axisOutBatch']
time_list = []

for idx in range(0, len(imgBatch)):

    img, speed, steer, out = np.expand_dims(imgBatch[idx], axis=0), np.expand_dims(speedInBatch[idx], axis=0), np.expand_dims(steerInBatch[idx], axis=0), np.expand_dims(axisOutBatch[idx], axis=0)

    print("img.shape: ", img.shape, "speed.shape: ", speed.shape, "steer.shape: ", steer.shape, "out.shape: ", out.shape)

    start_time = time.time()
    prediction = model.predict([img, speed, steer])
    end_time = time.time()
    dt= end_time-start_time
    print("Time taken:", dt,'sec')
    time_list.append(dt)

    print(prediction)
    print(out)
print("time_list:",time_list)
print("Average time taken:", sum(time_list[5:])/(len(time_list)-5),'sec')
