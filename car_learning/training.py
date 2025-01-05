# import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from sklearn.utils import shuffle

#### STEP 1 - INITIALIZE DATA
#### STEP 2 - VISUALIZE AND BALANCE DATA
#### STEP 3 - PREPARE FOR PROCESSING
#### STEP 4 - SPLIT FOR TRAINING AND VALIDATION
#### STEP 5 - AUGMENT DATA
#### STEP 6 - PREPROCESS
#### STEP 7 - CREATE MODEL
#### STEP 8 - TRAIN
#### STEP 9 - SAVE THE MODEL
#### STEP 10 - PLOT THE RESULTS

# create pandas dataframe from /data/data.txt
df = pd.read_csv('/data/data.txt', sep=',', names=['steering', 'forward', 'backward', 'boost', 'recording', 'exit', 'timestep', 'idx'])


# create tf dataloader where in each epoch we generate random suffle of indicies of dataframe rows
# then we load images and steering angles from the dataframe and return them in batch

def loader(df, batch_size=32):
    num_samples = len(df)
    while True: # Loop forever so the generator never terminates
        df = df.sample(frac=1).reset_index(drop=True)
        for offset in range(0, num_samples, batch_size):
            batch_samples = df.iloc[offset:offset+batch_size]

            images = []
            steering = []
            # F:\AA-Private\car\all_data\data_good\000001.png
            for index, batch_sample in batch_samples.iterrows():
                # make 6 digits from batch_sample['idx'] and end with png
                img = cv2.imread('F:/AA-Private/car/all_data/data_good/' + str(batch_sample['idx']).zfill(6) + '.png')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                steering_angle = batch_sample['steering']
                images.append(img)
                steering.append(steering_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(steering)
            yield shuffle(X_train, y_train)

if __name__ == '__main__':
    # try the data loader
    loader = loader(df)
    X, y = next(loader)
    print(X.shape, y.shape)

